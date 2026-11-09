import math

import torch
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import ModelConfig
from .layer import Block


class Model(PreTrainedModel, GenerationMixin):
    """Decoder-only LM with optional multi-token prediction (MTP).

    With `n_future_tokens == 1` this is a standard next-token model with
    `num_hidden_layers` blocks. With `n_future_tokens = n > 1` it follows
    Gloeckle et al. (2024): a shared trunk of `num_hidden_layers - 1` blocks
    feeds `n` parallel single-block heads, where head `i` predicts token
    `t + 1 + i`. All heads share the tied unembedding. Head 0 is the
    next-token head, so the next-token path is still `num_hidden_layers`
    blocks deep.
    """
    config_class = ModelConfig
    base_model_prefix = "modern_transformer"
    _tied_weights_keys = {"lm_head.weight": "tok_emb_table.weight"}  # noqa: RUF012

    def __init__(
            self,
            config: ModelConfig,
    ) -> None:
        super().__init__(config)
        self.n_future_tokens = config.n_future_tokens
        if self.n_future_tokens < 1:
            raise ValueError("n_future_tokens must be >= 1")
        if self.n_future_tokens > 1 and config.num_hidden_layers < 2:
            raise ValueError("MTP needs num_hidden_layers >= 2 (trunk + head)")

        self.tok_emb_table = nn.Embedding(
            config.vocab_size, config.embedding_size)
        self.emb_drop = nn.Dropout(config.dropout)
        self.rotary_emb = RotaryEmbedding(dim=config.rope_size)

        n_trunk = config.num_hidden_layers if self.n_future_tokens == 1\
            else config.num_hidden_layers - 1
        self.layers = nn.ModuleList(
            [self._make_block(config) for _ in range(n_trunk)])
        if self.n_future_tokens == 1:
            self.rms_norm_f = nn.RMSNorm(config.embedding_size, eps=1e-6)
        else:
            self.mtp_heads = nn.ModuleList([
                self._make_block(config)
                for _ in range(self.n_future_tokens)])
            self.mtp_norms = nn.ModuleList([
                nn.RMSNorm(config.embedding_size, eps=1e-6)
                for _ in range(self.n_future_tokens)])
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.block_size = config.block_size

        self.config.tie_word_embeddings = True
        self.post_init()

        self.last_lm_loss = None
        self.last_router_loss = None
        self.last_mtp_losses = None

    @staticmethod
    def _make_block(config: ModelConfig) -> Block:
        return Block(
            embedding_size=config.embedding_size,
            sa_head_size=config.sa_head_size,
            ff_hidden_size=config.ff_hidden_size,
            n_experts=config.experts,
            n_active_experts=config.active_experts,
            rope_size=config.rope_size,
            kv_latent_size=config.kv_latent_size,
            num_attn_heads=config.num_attention_heads,
            num_residual_streams=config.num_residual_streams,
            block_size=config.block_size,
            activation=config.non_linearity,
            dropout=config.dropout,
        )

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            if (
                any(x in type(module).__name__ for x in ["MHCRouter"])
                or (
                    hasattr(module, 'out_features')
                    and
                    module.out_features in [4, 16]
            )):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    if module.out_features == 16: # TODO: make more general
                        with torch.no_grad():
                            module.bias.copy_(torch.eye(4).view(-1) * 0.1)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=1.0 / math.sqrt(module.embedding_dim))
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_emb_table

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.tok_emb_table = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        **kwargs
    ):
        past_length = 0

        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()

        if past_length > 0:
            input_ids = input_ids[:, -1:]
        else:
            input_ids = input_ids[:, -self.block_size:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    # ------------------------------------------------------------------ #
    # Building blocks of the forward pass
    # ------------------------------------------------------------------ #

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb_drop(self.tok_emb_table(input_ids))
        x = x.to(next(self.parameters()).dtype)
        if self.config.num_residual_streams > 1:
            x = x.unsqueeze(2).repeat(
                1, 1, self.config.num_residual_streams, 1)
        return x

    def _collapse_streams(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2) if self.config.num_residual_streams > 1 else x

    def _run_trunk(
        self,
        x: torch.Tensor,
        past_key_values: Cache | None,
        past_length: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        aux_losses = []
        for i, layer in enumerate(self.layers):
            x, router_loss = layer(
                x,
                self.rotary_emb,
                past_key_values=past_key_values,
                past_length=past_length,
                layer_idx=i,
            )
            if router_loss is not None:
                aux_losses.append(router_loss)
        return x, aux_losses

    def _run_head(
        self,
        head_idx: int,
        x: torch.Tensor,
        past_key_values: Cache | None,
        past_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Trunk output -> normalized hidden state [B, T, C] for one head."""
        if self.n_future_tokens == 1:
            return self.rms_norm_f(self._collapse_streams(x)), None
        y, router_loss = self.mtp_heads[head_idx](
            x,
            self.rotary_emb,
            past_key_values=past_key_values,
            past_length=past_length,
            layer_idx=len(self.layers) + head_idx,
        )
        return self.mtp_norms[head_idx](self._collapse_streams(y)), router_loss

    def _head_loss(
        self,
        head_idx: int,
        x: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed CE of one head, plus its MoE aux loss (0 if dense).

        Runs under activation checkpointing during MTP training, so the
        [B, T, V] logits of only one head are alive at any time.
        """
        h, router_loss = self._run_head(head_idx, x, None, 0)
        logits = self.lm_head(h)
        ce = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        if router_loss is None:
            router_loss = torch.zeros((), device=ce.device, dtype=ce.dtype)
        return ce, router_loss

    def _mtp_training_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        aux_losses: list[torch.Tensor],
        num_items_in_batch: int | None,
    ) -> torch.Tensor:
        B = labels.shape[0]
        n_valid_0 = (labels != -100).sum().clamp_min(1)
        head_losses, logged_losses = [], []
        for i in range(self.n_future_tokens):
            # labels[:, t] is token t+1; head i predicts token t+1+i.
            targets = labels if i == 0 else torch.cat(
                [labels[:, i:], labels.new_full((B, i), -100)], dim=1)
            if self.config.mtp_checkpoint:
                ce, router_loss = checkpoint(
                    self._head_loss, i, x, targets, use_reentrant=False)
            else:
                ce, router_loss = self._head_loss(i, x, targets)
            # Head i has i fewer targets per sequence; scale the trainer's
            # token count (num_items_in_batch, over all accumulation steps)
            # accordingly so every head is a true per-token mean.
            n_valid = (targets != -100).sum().clamp_min(1)
            denom = n_valid if num_items_in_batch is None\
                else num_items_in_batch * n_valid / n_valid_0
            head_losses.append(ce / denom)
            logged_losses.append(ce.detach() / n_valid)
            if self.config.experts > 1:
                aux_losses.append(router_loss)

        head_losses = torch.stack(head_losses)
        lm_loss = head_losses[0]
        loss = lm_loss + self.config.mtp_loss_weight * head_losses[1:].mean()
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        if aux_loss is not None:
            loss = loss + self.config.router_loss_coef * aux_loss

        # Logged values are per-token means of this micro-batch, independent
        # of gradient-accumulation scaling.
        self.last_mtp_losses = torch.stack(logged_losses)
        self.last_lm_loss = self.last_mtp_losses[0]
        self.last_router_loss = aux_loss.detach()\
            if aux_loss is not None else None
        return loss

    # ------------------------------------------------------------------ #
    # Public forward passes
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        num_items_in_batch: int | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        return_per_sample_loss: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Standard HF forward.

        Logits and eval losses come from head 0 (next token) only, so eval
        perplexity and HellaSwag stay comparable to non-MTP models. The extra
        heads are only run for the MTP training loss and by
        `forward_all_heads` (speculative decoding).
        """
        use_cache = use_cache if use_cache is not None\
            else getattr(self.config, "use_cache", True)
        cache = past_key_values if use_cache else None

        if labels is not None and attention_mask is not None:
            labels[attention_mask == 0] = -100

        x = self._embed(input_ids)
        past_length = past_key_values.get_seq_length()\
            if past_key_values is not None else 0
        x, aux_losses = self._run_trunk(x, cache, past_length)

        if (
            labels is not None
            and self.training
            and self.n_future_tokens > 1
            and not return_per_sample_loss
        ):
            loss = self._mtp_training_loss(
                x, labels, aux_losses, num_items_in_batch)
            return CausalLMOutputWithPast(loss=loss)  # type: ignore

        h, router_loss = self._run_head(0, x, cache, past_length)
        if router_loss is not None:
            aux_losses.append(router_loss)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        logits = self.lm_head(h)

        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape

            lm_loss = per_token = F.cross_entropy(
                logits.view(-1, C),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )

            if return_per_sample_loss:
                lm_loss = lm_loss.view(B, T)
                loss = lm_loss.sum(dim=1) / (labels != -100).sum(dim=1).clamp_min(1)
            elif num_items_in_batch is not None:
                loss = lm_loss = lm_loss.sum() / num_items_in_batch
                if aux_loss is not None:
                    loss = lm_loss + self.config.router_loss_coef * aux_loss
            else:
                loss = lm_loss = lm_loss.mean()
                if aux_loss is not None:
                    loss = lm_loss + self.config.router_loss_coef * aux_loss

            # Per-token mean of this micro-batch (not scaled by grad accum)
            self.last_lm_loss = per_token.detach().sum()\
                / (labels != -100).sum().clamp_min(1)
            self.last_router_loss = aux_loss.detach()\
                if aux_loss is not None else None

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss, # type: ignore
            past_key_values=past_key_values if use_cache else None, # type: ignore
        )

    @torch.no_grad()
    def forward_all_heads(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Logits of every head, shape [n_future_tokens, B, T, V].

        `logits[i, :, t]` predicts token t+1+i. Every head writes its own
        KV-cache layer, so all heads must be run on every position for the
        cache to stay consistent (used by speculative decoding).
        """
        x = self._embed(input_ids)
        past_length = past_key_values.get_seq_length()\
            if past_key_values is not None else 0
        x, _ = self._run_trunk(x, past_key_values, past_length)
        return torch.stack([
            self.lm_head(self._run_head(i, x, past_key_values, past_length)[0])
            for i in range(self.n_future_tokens)
        ])

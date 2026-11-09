import copy
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
from .layer import Block, prefix_rms_norm


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
        for frac in config.matryoshka_fracs:
            config.matryoshka_dims(frac)  # validates every nested size
        if len(config.matryoshka_fracs) > 1 and config.num_residual_streams > 1:
            raise NotImplementedError("Matryoshka nesting with mHC streams")
        # Matryoshka width used by forward() when no `matryoshka_frac` is
        # passed (the trainer sets it per micro-batch and per eval pass).
        self.active_frac = 1.0

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
            if config.draft_vocab_size > 0:
                if config.draft_vocab_size > config.vocab_size:
                    raise ValueError("draft_vocab_size > vocab_size")
                # Persistent so checkpoints carry their draft vocabulary.
                # Placeholder until set_draft_vocab() is called (train.py).
                self.register_buffer(
                    "draft_vocab", torch.arange(config.draft_vocab_size))
                self.register_buffer(
                    "vocab_to_draft", torch.full((config.vocab_size,), -100))
                self.vocab_to_draft[:config.draft_vocab_size] =\
                    torch.arange(config.draft_vocab_size)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.block_size = config.block_size

        self.config.tie_word_embeddings = True
        self.post_init()

        self.last_lm_loss = None
        self.last_router_loss = None
        self.last_mtp_losses = None

    @torch.no_grad()
    def set_draft_vocab(self, token_ids: torch.Tensor) -> None:
        """Set the reduced vocabulary of the extra MTP heads (any order)."""
        token_ids = torch.as_tensor(token_ids).long().unique()  # sorted
        if token_ids.numel() != self.config.draft_vocab_size:
            raise ValueError(
                f"expected {self.config.draft_vocab_size} unique ids, "
                f"got {token_ids.numel()}")
        if token_ids.min() < 0 or token_ids.max() >= self.config.vocab_size:
            raise ValueError("draft token id out of range")
        token_ids = token_ids.to(self.draft_vocab.device)
        self.draft_vocab.copy_(token_ids)
        self.vocab_to_draft.fill_(-100)
        self.vocab_to_draft[token_ids] = torch.arange(
            token_ids.numel(), device=token_ids.device)

    def _uses_draft_vocab(self, head_idx: int) -> bool:
        return head_idx > 0 and self.config.draft_vocab_size > 0

    def _head_logits(self, head_idx: int, h: torch.Tensor) -> torch.Tensor:
        """Unembed one head. Extra heads with a draft vocab return logits
        over `draft_vocab` (indices into it, not token ids). A Matryoshka
        sub-width hidden state (d' = h.shape[-1]) uses the first d' columns
        of the tied unembedding."""
        d = h.shape[-1]
        if not self._uses_draft_vocab(head_idx):
            if d == self.config.embedding_size:
                return self.lm_head(h)
            return F.linear(h, self.lm_head.weight[:, :d], self.lm_head.bias)
        return F.linear(
            h,
            self.lm_head.weight[self.draft_vocab, :d],
            self.lm_head.bias[self.draft_vocab],
        )

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
            exact_ffn_hidden=config.ff_hidden_exact,
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

    def _embed(self, input_ids: torch.Tensor, frac: float) -> torch.Tensor:
        x = self.tok_emb_table(input_ids)
        if frac < 1.0:  # Matryoshka: residual stream = first d' dims
            x = x[..., :round(self.config.embedding_size * frac)]
        x = self.emb_drop(x)
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
        frac: float,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        aux_losses = []
        for i, layer in enumerate(self.layers):
            x, router_loss = layer(
                x,
                self.rotary_emb,
                past_key_values=past_key_values,
                past_length=past_length,
                layer_idx=i,
                frac=frac,
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
        frac: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Trunk output -> normalized hidden state [B, T, C] for one head."""
        if self.n_future_tokens == 1:
            return prefix_rms_norm(
                self.rms_norm_f, self._collapse_streams(x)), None
        y, router_loss = self.mtp_heads[head_idx](
            x,
            self.rotary_emb,
            past_key_values=past_key_values,
            past_length=past_length,
            layer_idx=len(self.layers) + head_idx,
            frac=frac,
        )
        return prefix_rms_norm(
            self.mtp_norms[head_idx], self._collapse_streams(y)), router_loss

    def _head_loss(
        self,
        head_idx: int,
        x: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor | None,
        frac: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Summed CE of one head, its MoE aux loss (0 if dense), and the
        number of targets that counted.

        The head block always runs on every position (attention needs
        them); only the unembedding + CE is restricted to `positions`
        (flattened B*T indices) and, for extra heads, to the draft vocab.
        Targets outside the draft vocab are ignored. Runs under activation
        checkpointing when `mtp_checkpoint` is set.
        """
        h, router_loss = self._run_head(head_idx, x, None, 0, frac)
        h = h.reshape(-1, h.shape[-1])
        targets = targets.reshape(-1)
        if positions is not None:
            h, targets = h[positions], targets[positions]
        if self._uses_draft_vocab(head_idx):
            mapped = self.vocab_to_draft[targets.clamp_min(0)]
            targets = torch.where(targets == -100, -100, mapped)
        ce = F.cross_entropy(
            self._head_logits(head_idx, h),
            targets,
            ignore_index=-100,
            reduction="sum",
        )
        if router_loss is None:
            router_loss = torch.zeros((), device=ce.device, dtype=ce.dtype)
        return ce, router_loss, (targets != -100).sum()

    def _mtp_training_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        aux_losses: list[torch.Tensor],
        num_items_in_batch: int | None,
        frac: float,
    ) -> torch.Tensor:
        B, T = labels.shape
        n_valid_0 = (labels != -100).sum().clamp_min(1)
        pos_frac = self.config.mtp_position_frac
        n_sampled = max(1, round(pos_frac * B * T))
        head_losses, logged_losses = [], []
        for i in range(self.n_future_tokens):
            # labels[:, t] is token t+1; head i predicts token t+1+i.
            targets = labels if i == 0 else torch.cat(
                [labels[:, i:], labels.new_full((B, i), -100)], dim=1)
            # Extra heads: fixed-size random subset of positions (fixed size
            # keeps shapes static for torch.compile). Drawn outside the
            # checkpointed function so the recompute sees the same subset.
            positions = torch.randperm(B * T, device=labels.device)[:n_sampled]\
                if i > 0 and pos_frac < 1.0 else None
            if self.config.mtp_checkpoint:
                ce, router_loss, n_valid = checkpoint(
                    self._head_loss, i, x, targets, positions, frac,
                    use_reentrant=False)
            else:
                ce, router_loss, n_valid = self._head_loss(
                    i, x, targets, positions, frac)
            # Extra heads see fewer targets (shifted ends, sampled positions,
            # out-of-draft-vocab targets); scale the trainer's token count
            # (num_items_in_batch, over all accumulation steps) accordingly
            # so every head is a per-counted-target mean.
            n_valid = n_valid.clamp_min(1)
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
        matryoshka_frac: float | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Standard HF forward.

        Logits and eval losses come from head 0 (next token) only, so eval
        perplexity and HellaSwag stay comparable to non-MTP models. The extra
        heads are only run for the MTP training loss and by
        `forward_all_heads` (speculative decoding).

        `matryoshka_frac` (default: `self.active_frac`) selects the nested
        width; a KV cache is only valid for the width it was built with.
        """
        frac = self.active_frac if matryoshka_frac is None else matryoshka_frac
        use_cache = use_cache if use_cache is not None\
            else getattr(self.config, "use_cache", True)
        cache = past_key_values if use_cache else None

        if labels is not None and attention_mask is not None:
            labels[attention_mask == 0] = -100

        x = self._embed(input_ids, frac)
        past_length = past_key_values.get_seq_length()\
            if past_key_values is not None else 0
        x, aux_losses = self._run_trunk(x, cache, past_length, frac)

        if (
            labels is not None
            and self.training
            and self.n_future_tokens > 1
            and not return_per_sample_loss
        ):
            loss = self._mtp_training_loss(
                x, labels, aux_losses, num_items_in_batch, frac)
            return CausalLMOutputWithPast(loss=loss)  # type: ignore

        h, router_loss = self._run_head(0, x, cache, past_length, frac)
        if router_loss is not None:
            aux_losses.append(router_loss)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        logits = self._head_logits(0, h)

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
    ) -> list[torch.Tensor]:
        """Logits of every head, each [B, T, V_i].

        `logits[i][:, t]` predicts token t+1+i. Head 0 is over the full
        vocab; extra heads are over `draft_vocab` when one is set (see
        `predict_all_heads` for token ids). Every head writes its own
        KV-cache layer, so all heads must be run on every position for the
        cache to stay consistent (used by speculative decoding).
        """
        frac = self.active_frac
        x = self._embed(input_ids, frac)
        past_length = past_key_values.get_seq_length()\
            if past_key_values is not None else 0
        x, _ = self._run_trunk(x, past_key_values, past_length, frac)
        return [
            self._head_logits(
                i, self._run_head(i, x, past_key_values, past_length, frac)[0])
            for i in range(self.n_future_tokens)
        ]

    @torch.no_grad()
    def predict_all_heads(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Greedy token ids of every head, shape [n_future_tokens, B, T]."""
        preds = []
        for i, logits in enumerate(
                self.forward_all_heads(input_ids, past_key_values)):
            ids = logits.argmax(-1)
            preds.append(self.draft_vocab[ids]
                         if self._uses_draft_vocab(i) else ids)
        return torch.stack(preds)

    @torch.no_grad()
    def extract_matryoshka(self, frac: float) -> "Model":
        """Standalone model equal to this one run at width `frac`.

        Every nested weight is sliced to its prefix (two-part weights,
        kv_down = [latent | rope] and MoE gate_up = [gate | up], are sliced
        per part and re-concatenated). Raises on any parameter it does not
        know how to slice, so new modules can't be silently mis-extracted.
        """
        if frac not in self.config.matryoshka_fracs:
            raise ValueError(f"{frac} is not one of the trained sizes "
                             f"{self.config.matryoshka_fracs}")
        dims = self.config.matryoshka_dims(frac)
        d, n_heads = dims["embedding_size"], dims["num_attention_heads"]
        latent, hidden = dims["kv_latent_size"], dims["ffn_hidden"]
        full = self.config.matryoshka_dims(1.0)
        latent_full, hidden_full = full["kv_latent_size"], full["ffn_hidden"]
        head_dim = self.config.sa_head_size
        nope_dim = head_dim - self.config.rope_size

        config = copy.deepcopy(self.config)
        config.embedding_size = d
        config.num_attention_heads = n_heads
        config.kv_latent_size = latent
        config.ff_hidden_size = hidden
        config.ff_hidden_exact = True
        config.matryoshka_fracs = [1.0]

        def slice_param(name: str, w: torch.Tensor) -> torch.Tensor:
            last = name.split(".")[-2:]
            if name in ("tok_emb_table.weight", "lm_head.weight") or\
                    name.endswith("router.weight"):
                return w[:, :d]
            if name in ("lm_head.bias", "rotary_emb.freqs",
                        "draft_vocab", "vocab_to_draft") or last[-1] == "mask":
                return w
            if last == ["q_proj", "weight"]:
                return w[:n_heads * head_dim, :d]
            if last == ["kv_down", "weight"]:
                return torch.cat([w[:latent], w[latent_full:]])[:, :d]
            if last == ["k_up", "weight"]:
                return w[:n_heads * nope_dim, :latent]
            if last == ["v_up", "weight"]:
                return w[:n_heads * head_dim, :latent]
            if last == ["proj", "weight"]:
                return w[:d, :n_heads * head_dim]
            if last in (["gate_proj", "weight"], ["up_proj", "weight"]):
                return w[:hidden, :d]
            if last == ["down_proj", "weight"]:
                return w[:d, :hidden]
            if last[-1] == "gate_up_weight":
                return torch.cat(
                    [w[:, :hidden], w[:, hidden_full:hidden_full + hidden]],
                    dim=1)[..., :d]
            if last[-1] == "up_weight":
                return w[:, :hidden, :d]
            if last[-1] == "down_weight":
                return w[:, :d, :hidden]
            if w.ndim == 1 and ("rms_norm" in name or "mtp_norms" in name):
                return w[:d]
            raise NotImplementedError(f"don't know how to slice {name}")

        param = next(self.parameters())
        small = Model(config).to(device=param.device, dtype=param.dtype)
        small.generation_config = copy.deepcopy(self.generation_config)
        state = {name: slice_param(name, w).clone()
                 for name, w in self.state_dict().items()}
        small.load_state_dict(state, strict=True)
        return small

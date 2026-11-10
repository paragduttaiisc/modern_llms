"""`LLMTrainer`: the HuggingFace `Trainer` customized for LLM pretraining.

HF's `Trainer` already handles the training loop, gradient accumulation,
mixed precision, distributed training (through Accelerate), checkpointing
and logging. This subclass replaces or extends the parts that are specific
to this project:

- the optimizer: Muon for the hidden 2-D/3-D weight matrices and AdamW for
  everything else, in one `muonium.Muon` optimizer (`create_optimizer`);
- the LR schedule: linear warmup, cosine decay, then a constant floor
  (`create_scheduler`);
- Matryoshka training: which nested width each micro-batch trains
  (`training_step`, `_sample_matryoshka_frac`);
- evaluation: eval loss of every Matryoshka size plus HellaSwag accuracy
  (`evaluate`, `evaluate_hellaswag`);
- logging: throughput, loss components, LRs and MoE expert statistics
  (`log`, `_log_window_losses`).
"""
import math
import os
import random
import time
from typing import Any, cast

import torch
from muonium import Muon
from torch.optim.adam import adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import Trainer

from .data_utils import HellaswagDataset, hellaswag_collate_fn
from .misc_utils import human_readable_numbers as hrn


class MuonFP32State(Muon):
    """Muon whose AdamW groups keep their moments in fp32.

    muonium already keeps Muon momentum in fp32, but its AdamW moments take
    the parameter dtype (bf16 here). The weights stay bf16: each AdamW step
    runs on fp32 copies of the weights and gradients, and the result is
    rounded back into the bf16 weights. Works on FSDP2 (DTensor) params too.

    Why: AdamW's second moment (a running mean of squared gradients) can
    span many orders of magnitude, and bf16's 8-bit mantissa loses small
    updates to it. The fp32 moments double the AdamW state memory, but only
    for the AdamW params (embeddings, norms, routers). Muon groups are
    untouched.
    """

    def _step_adam_group(self, group: dict[str, Any]) -> None:
        """One AdamW step for `group`, overriding muonium's version.

        Gathers the params that have a gradient, lazily creates fp32
        `exp_avg` / `exp_avg_sq` state, runs `torch.optim.adam`'s functional
        foreach kernel on fp32 copies of weights and gradients, and copies
        the updated fp32 weights back into the (bf16) parameters in place.
        """
        params, params32, grads32, exp_avgs, exp_avg_sqs, steps =\
            [], [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            for key in ("exp_avg", "exp_avg_sq"):  # older bf16 checkpoints
                if state[key].dtype != torch.float32:
                    state[key] = state[key].float()
            # Loading a checkpoint can map the step counter onto the GPU;
            # the non-capturable `adam` path expects CPU step tensors.
            if state["step"].device.type != "cpu":
                state["step"] = state["step"].cpu()
            params.append(p)
            params32.append(p.detach().float())
            grads32.append(p.grad.float())
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            steps.append(state["step"])
        if not params:
            return
        # lr / wd can be tensors in the group; `adam` is given floats.
        lr, wd = group["lr"], group["wd"]
        beta1, beta2 = group["betas"]
        adam(
            params32, grads32, exp_avgs, exp_avg_sqs, [], steps,
            amsgrad=False, has_complex=False, beta1=beta1, beta2=beta2,
            lr=float(lr) if torch.is_tensor(lr) else lr,
            weight_decay=float(wd) if torch.is_tensor(wd) else wd,
            eps=group["eps"], maximize=group["maximize"], capturable=False,
            differentiable=False, fused=False, foreach=True,
            decoupled_weight_decay=group["decoupled_weight_decay"],
        )
        with torch.no_grad():  # (DTensor has no _foreach_copy_)
            for p, p32 in zip(params, params32):
                p.copy_(p32)


class LLMTrainer(Trainer):
    """HF `Trainer` with Muon/AdamW, a cosine schedule, Matryoshka training,
    HellaSwag eval and extra logging (see the module docstring).

    Used with `Model` from `model/model.py` and `TokenDataset`. Beyond the
    standard `Trainer` arguments it takes:

    Args:
        block_size: sequence length, used to turn tokens/s into samples/s.
        warmup_iters: optimizer steps of linear LR warmup.
        last_decay_iter: step where the cosine decay reaches 0.1x the peak
            LR; the LR stays there afterwards.
        muon_lr: peak LR of the Muon groups. The AdamW groups use
            `args.learning_rate`; both follow the same schedule multiplier.
        muon_weight_decay: weight decay of the Muon groups. The decayed
            AdamW group uses `args.weight_decay`.
        matryoshka_full_prob: probability that a micro-batch trains the full
            width (only with more than one `matryoshka_fracs`).
        matryoshka_full_per_step: make the first micro-batch of every
            optimizer step full width (see `_sample_matryoshka_frac`).
        matryoshka_all_sizes: train every width on every micro-batch instead
            of sampling one.
    """

    def __init__(
            self,
            *args,
            block_size: int = 4096,
            warmup_iters: int = 1000,
            last_decay_iter: int = 50000,
            muon_lr: float = 0.02,
            muon_weight_decay: float = 0.01,
            matryoshka_full_prob: float = 0.5,
            matryoshka_full_per_step: bool = False,
            matryoshka_all_sizes: bool = False,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # Matryoshka: each micro-batch trains one nested width, the full
        # model with probability `matryoshka_full_prob`, otherwise a smaller
        # size uniformly. A dedicated RNG seeded identically on every rank
        # keeps all ranks on the same size (so no rank waits on a bigger one).
        self.matryoshka_fracs = list(getattr(
            self._llm.config, "matryoshka_fracs", [1.0]))
        self.matryoshka_full_prob = matryoshka_full_prob
        self._matryoshka_rng = random.Random(self.args.seed)
        # Every optimizer step contains a full-size micro-batch (the first);
        # the rest are sampled so the overall full share stays at full_prob.
        # Without it, with 2 micro-batches per step a quarter of the steps
        # have no full-size gradient, and training diverges at muon_lr 0.02.
        self.matryoshka_full_per_step = matryoshka_full_per_step
        self._micro_batch_idx = 0
        # Alternative to sampling: every micro-batch trains every size on
        # the same tokens (loss = sum over sizes), so the full model sees
        # exactly the data and gradient scale of a run without Matryoshka.
        self.matryoshka_all_sizes =\
            matryoshka_all_sizes and len(self.matryoshka_fracs) > 1
        if self.matryoshka_all_sizes and matryoshka_full_per_step:
            raise ValueError("matryoshka_all_sizes trains every size each "
                             "micro-batch; don't combine it with "
                             "matryoshka_full_per_step")
        # Per-size sums of per-token losses since the last log (see log())
        self._loss_window: dict[float, dict[str, Any]] = {}

        self.block_size = block_size
        self.warmup_iters = warmup_iters
        self.last_decay_iter = last_decay_iter

        self.muon_lr = muon_lr
        self.muon_weight_decay = muon_weight_decay

        self.total_tokens = 0
        self.log_start_time = time.time()
        self.log_start_tokens = 0
        self.log_start_steps = 0

    @property
    def _llm(self) -> Any:
        """`self.model`, typed loosely for the type checker.

        HF types `Trainer.model` as an optional generic module; this trainer
        always holds this repo's `Model`, whose config and logging
        attributes are read below.
        """
        return self.model

    def create_optimizer(
            self, model: torch.nn.Module | None = None,
    ) -> torch.optim.Optimizer:
        """Build one Muon optimizer whose param groups use Muon or AdamW.

        Muon (MomentUm Orthogonalized by Newton-Schulz) replaces each 2-D
        weight's momentum-averaged gradient by its nearest semi-orthogonal
        matrix, an update with all singular values equal. It suits hidden
        weight matrices; embeddings, the LM head (tied to the embeddings),
        routers, mHC projections, norm gains and biases keep AdamW.

        Params are routed by name and shape, so a new module must be checked
        against these rules (the summary printed on rank 0 shows the split):
          - 3-D params (stacked MoE expert weights `[E, out, in]`): Muon with
            `flatten=False`, i.e. each expert's matrix orthogonalized alone;
          - `q_proj` / `k_up` / `v_up` weights: head-wise Muon, the rows
            split into one block per attention head (`split_sizes`) and each
            block orthogonalized separately;
          - other 2-D weights: Muon, unless the name contains `router`,
            `mhc`, `emb` or `lm_head`;
          - everything else: AdamW, with weight decay for >= 2-D params and
            without for 1-D params (norm gains, biases).

        `model` is the prepared model HF passes when it creates the optimizer
        after wrapping the model; otherwise `self.model` is used. Returns the
        existing optimizer if one was already created.
        """
        if self.optimizer is not None:
            return self.optimizer

        muon_2d_params = []
        muon_q_params = []
        muon_k_up_params = []
        muon_v_up_params = []
        muon_3d_params = []
        adamw_decay_params = []
        adamw_nodecay_params = []

        n_heads = self._llm.config.num_attention_heads
        head_dim = self._llm.config.sa_head_size
        rope_dim = self._llm.config.rope_size
        nope_dim = head_dim - rope_dim

        # Row counts per head of each attention up-projection. `q_proj` rows
        # are [head 0 | head 1 | ...], each head_dim wide (nope + rope); `k_up`
        # only produces the nope part of K (the rope part comes from
        # `kv_down`); `v_up` gives full-width V heads.
        q_split_sizes = (head_dim,) * n_heads
        k_up_split_sizes = (nope_dim,) * n_heads
        v_up_split_sizes = (head_dim,) * n_heads

        # HF passes the prepared (possibly wrapped) model when it delays
        # optimizer creation, e.g. for some FSDP setups. Parameter names keep
        # their substrings either way, so the routing rules still apply.
        opt_model = model if model is not None else self._llm
        for name, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue

            is_router = "router" in name.lower()
            is_mhc = "mhc" in name.lower()
            is_embedding = "emb" in name.lower()
            # Match only the LM head; a bare "head" also matches the
            # attention module name "sa_heads".
            is_head = "lm_head" in name.lower()

            # Stacked MoE expert weights [E, out, in]: gate_up_weight,
            # up_weight (GELU/SqReLU), down_weight
            if p.ndim == 3:
                muon_3d_params.append(p)
                continue
            if p.ndim == 2 and name.lower().endswith("q_proj.weight"):
                muon_q_params.append(p)
                continue
            if p.ndim == 2 and name.lower().endswith("k_up.weight"):
                muon_k_up_params.append(p)
                continue
            if p.ndim == 2 and name.lower().endswith("v_up.weight"):
                muon_v_up_params.append(p)
                continue
            if (
                p.ndim == 2
                and not is_router
                and not is_mhc
                and not is_embedding
                and not is_head
            ):
                muon_2d_params.append(p)
                continue
            if p.ndim >= 2:
                adamw_decay_params.append(p)
            else:
                adamw_nodecay_params.append(p)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:  # noqa: PLW1508
            def sum_p(params):
                return sum(p.numel() for p in params)

            print(
                f"Muon (2D): {len(muon_2d_params)} tensors, "
                f"{hrn(sum_p(muon_2d_params))} params\n"
                f"Muon (Q head-wise): {len(muon_q_params)} tensors, "
                f"{hrn(sum_p(muon_q_params))} params\n"
                f"Muon (K Up): {len(muon_k_up_params)} tensors, "
                f"{hrn(sum_p(muon_k_up_params))} params\n"
                f"Muon (V Up): {len(muon_v_up_params)} tensors, "
                f"{hrn(sum_p(muon_v_up_params))} params\n"
                f"Muon (3D): {len(muon_3d_params)} tensors, "
                f"{hrn(sum_p(muon_3d_params))} params\n"
                f"AdamW (Decay): {len(adamw_decay_params)} tensors, "
                f"{hrn(sum_p(adamw_decay_params))} params\n"
                f"AdamW (No Decay): {len(adamw_nodecay_params)} tensors, "
                f"{hrn(sum_p(adamw_nodecay_params))} params")

        # Muon groups take `lr` / `wd` from the constructor defaults below;
        # AdamW groups override them with the HF TrainingArguments values.
        self.optimizer = MuonFP32State([{
            "params": muon_2d_params,
            "algorithm": "muon",
        }, {
            "params": muon_q_params,
            "algorithm": "muon",
            "split_sizes": q_split_sizes,
        }, {
            "params": muon_k_up_params,
            "algorithm": "muon",
            "split_sizes": k_up_split_sizes,
        }, {
            "params": muon_v_up_params,
            "algorithm": "muon",
            "split_sizes": v_up_split_sizes,
        }, {
            "params": muon_3d_params,
            "algorithm": "muon",
            "flatten": False,
        }, {
            "params": adamw_nodecay_params,
            "algorithm": "adamw",
            "lr": self.args.learning_rate,
            # muonium reads "wd"; a "weight_decay" key would be ignored and
            # the group would get the Muon default (muon_weight_decay).
            "wd": 0.0,
        }, {
            "params": adamw_decay_params,
            "algorithm": "adamw",
            "lr": self.args.learning_rate,
            "wd": self.args.weight_decay,
        }], lr=self.muon_lr, wd=self.muon_weight_decay,
        orthogonalization_strategy="newton_schulz", use_cautious_wd=False,
        adam_betas=(self.args.adam_beta1, self.args.adam_beta2),
        adam_eps=self.args.adam_epsilon, is_adamw=True, foreach=None)

        return self.optimizer

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Linear warmup, cosine decay to 0.1x, then constant 0.1x.

        One `LambdaLR` multiplier is shared by every param group, so Muon
        and AdamW groups keep the ratio of their peak LRs throughout:
          - steps [0, warmup_iters): linear from ~0 to 1;
          - steps [warmup_iters, last_decay_iter): cosine from 1 to 0.1;
          - afterwards: 0.1.
        `num_training_steps` (HF's argument) is not used; the schedule is
        set by `warmup_iters` and `last_decay_iter` alone.
        """
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = optimizer if optimizer is not None else self.optimizer

        warmup_iters = self.warmup_iters
        decay_end = self.last_decay_iter

        def lr_lambda(step: int) -> float:
            if step < warmup_iters:
                if warmup_iters == 0:
                    return 1.0
                progress = step / warmup_iters
                # Starts at 1e-6 rather than 0 so step 0 is not a no-op
                return 1e-6 + (1.0 - 1e-6) * progress
            if step < decay_end:
                decay_progress =\
                    (step - warmup_iters) / (decay_end - warmup_iters)
                cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                return 0.1 + 0.9 * cosine
            return 0.1

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return self.lr_scheduler

    def _sample_matryoshka_frac(self) -> float:
        """Width fraction for the next training micro-batch.

        Returns 1.0 (full model) with probability `matryoshka_full_prob`,
        otherwise one of the smaller fractions uniformly at random. Called
        once per micro-batch on every rank; the shared seed and identical
        call order give every rank the same sequence of sizes.

        With `matryoshka_full_per_step`, micro-batch 0 of each optimizer
        step (of k = gradient_accumulation_steps) is always full size, and
        the other k-1 are full with probability p' chosen so the overall
        share is unchanged: (1 + (k-1) p') / k = p, i.e.
        p' = (p k - 1) / (k - 1), clipped at 0. With k = 1 every micro-batch
        is full size.
        """
        smaller = self.matryoshka_fracs[1:]  # sorted descending, [0] == 1.0
        if not smaller:
            return 1.0
        full_prob = self.matryoshka_full_prob
        if self.matryoshka_full_per_step:
            k = self.args.gradient_accumulation_steps
            idx = self._micro_batch_idx % k
            self._micro_batch_idx += 1
            if idx == 0:
                return 1.0
            full_prob = max(0.0, (full_prob * k - 1) / (k - 1))
        if self._matryoshka_rng.random() < full_prob:
            return 1.0
        return self._matryoshka_rng.choice(smaller)

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """One micro-batch: forward, backward and loss bookkeeping.

        Sets which Matryoshka width(s) the model trains on this micro-batch,
        runs HF's `training_step` (forward + backward), and always resets the
        model to full width afterwards, since eval and generation use
        `model.active_frac`. Then adds the model's per-token loss components
        (`last_lm_loss`, `last_mtp_losses`) to the logging window and counts
        the tokens seen by all ranks for the tokens/s metric.

        Returns HF's loss for this micro-batch (scaled for gradient
        accumulation, as HF expects).
        """
        if self.matryoshka_all_sizes:
            frac = 1.0
            self._llm.train_fracs = self.matryoshka_fracs
        else:
            frac = self._sample_matryoshka_frac()
            self._llm.active_frac = frac
        try:
            loss = super().training_step(model, inputs, *args, **kwargs)
        finally:
            self._llm.active_frac = 1.0  # (eval/generation)
            self._llm.train_fracs = []
        # Sum the per-token losses of this micro-batch into the current
        # logging window (detached tensors: no GPU sync here).
        if self.matryoshka_all_sizes:
            per_size = self._llm.last_losses_by_frac
        else:
            mtp = getattr(self._llm, "last_mtp_losses", None)\
                if getattr(self._llm.config, "n_future_tokens", 1) > 1\
                else None
            per_size = {frac: (getattr(self._llm, "last_lm_loss", None), mtp)}
        for size, (lm, mtp) in per_size.items():
            window = self._loss_window.setdefault(
                size, {"n": 0, "lm": None, "mtp": None})
            window["n"] += 1
            for key, value in (("lm", lm), ("mtp", mtp)):
                if value is not None:
                    window[key] = value if window[key] is None\
                        else window[key] + value

        # Assumes every rank gets a batch of the same size, so the global
        # token count is the local one times the world size.
        if "input_ids" in inputs:
            local_tokens = inputs["input_ids"].numel()
            world_size =\
                self.args.world_size if hasattr(self.args, "world_size") else 1
            self.total_tokens += local_tokens * world_size
        return loss

    @torch.no_grad()
    def evaluate_hellaswag(self) -> float:
        """Zero-shot HellaSwag accuracy over the whole tokenized val set.

        Each example has 4 candidate sequences (shared context + one ending),
        with the correct one first. The model scores each candidate by its
        mean per-token loss (`return_per_sample_loss=True`), and an example
        counts as correct when candidate 0 has the lowest loss. The loader
        is sharded across ranks by Accelerate and the counts are summed over
        ranks, so every rank returns the same accuracy. Runs at the model's
        current `active_frac` (full size when called from `evaluate`).
        """
        dataset = HellaswagDataset(
            "data/Hellaswag/hellaswag_tokenized.npy"
        )

        loader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            persistent_workers=self.args.dataloader_persistent_workers,
            pin_memory=self.args.dataloader_pin_memory,
            collate_fn=hellaswag_collate_fn,
            shuffle=False,
        )

        loader = self.accelerator.prepare(loader)

        local_correct = 0
        local_total = 0

        for batch in loader:
            batch = {
                k: v.to(self.accelerator.device, non_blocking=True)
                for k, v in batch.items()
            }

            outputs = self._llm(**batch, return_per_sample_loss=True)

            # outputs.loss: [B * 4] per-candidate losses -> [B, 4]
            preds = outputs.loss.reshape(-1, 4).argmin(dim=1)

            local_correct += (preds == 0).sum().item()
            local_total += preds.numel()

        correct = torch.tensor([local_correct], device=self.accelerator.device)
        total = torch.tensor([local_total], device=self.accelerator.device)

        correct = cast(torch.Tensor, self.accelerator.gather(correct)).sum()
        total = cast(torch.Tensor, self.accelerator.gather(total)).sum()

        return (correct.float() / total.float()).item()

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """HF eval loss for every Matryoshka size, plus HellaSwag accuracy.

        Runs HF's `evaluate` at full width (keys `eval_loss`, ...), then once
        per smaller width with keys `eval_f<size>_*`, then HellaSwag on the
        full model (`eval_hellaswag_acc`, also logged). The model is left at
        full width in train mode, and the CUDA cache is emptied at the end.
        Note that in eval mode `Model` uses only head 0, so eval losses of
        MTP runs are comparable to dense next-token baselines.

        Returns the merged metrics dict.
        """
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # Eval loss of every smaller Matryoshka size, logged as
        # eval_f<size>_loss / eval_f<size>_perplexity
        for frac in self.matryoshka_fracs[1:]:
            self._llm.active_frac = frac
            try:
                metrics.update(super().evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_f{frac:g}",
                ))
            finally:
                self._llm.active_frac = 1.0

        self._llm.eval()
        hellaswag_acc = self.evaluate_hellaswag()
        self._llm.train()

        # Eval allocates differently shaped buffers (full-vocab logits of the
        # eval batch) than training. Release them so the cached-but-unused
        # blocks don't stay reserved on top of the training working set.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics["eval_hellaswag_acc"] = hellaswag_acc
        self.log({"eval_hellaswag_acc": hellaswag_acc})

        return metrics

    def _log_window_losses(self, logs: dict[str, Any]) -> None:
        """Log the components of HF's `loss` next to it (like router_loss).

        `loss` (HF) stays the optimized objective averaged over the logging
        window: it includes the extra MTP heads, the MoE router loss, and,
        with Matryoshka, every size. Added, as window means of per-token
        losses:
          - `lm_loss`: the full model's next-token loss (head 1)
          - `mtp_loss`: unweighted mean of the extra heads 2..n, so
            loss ~= lm_loss + mtp_loss_weight * mtp_loss
                    + router_loss_coef * router_loss
          - `mtp_loss_<k>`: head k alone (predicts token t+k)
          - `_f<size>` variants of all of these for smaller Matryoshka sizes
        Window means are averaged over all ranks, like HF's loss. HF calls
        log() with "loss" on every rank (right after its own all-gather), and
        every rank trains the same sizes in the same order, so the
        reductions below line up across ranks.
        """
        window, self._loss_window = self._loss_window, {}
        for frac, sums in sorted(window.items(), reverse=True):
            if sums["lm"] is None:
                continue
            suffix = "" if frac == 1.0 else f"_f{frac:g}"
            # [next-token, head 1..n] means of this window, then over ranks
            means = sums["lm"].reshape(1) if sums["mtp"] is None\
                else torch.cat([sums["lm"].reshape(1), sums["mtp"]])
            means = cast(torch.Tensor, self.accelerator.reduce(
                means.float() / sums["n"], reduction="mean")).tolist()
            logs[f"lm_loss{suffix}"] = means[0]
            if len(means) > 2:
                logs[f"mtp_loss{suffix}"] = sum(means[2:]) / len(means[2:])
                for i, head_loss in enumerate(means[1:]):
                    logs[f"mtp_loss_{i + 1}{suffix}"] = head_loss

    def log(self, logs: dict[str, Any], *args, **kwargs) -> None:
        """Add project metrics to HF's logs before they are reported.

        On training logs (those with "loss"): the per-window loss components
        (`_log_window_losses`), throughput since the previous log (tokens/s,
        iters/s, iter time, samples/s), the router loss and per-expert MoE
        stats of the last micro-batch, the current Muon and AdamW LRs, and
        `perplexity` = exp(lm_loss). For every `eval*_loss` key, the matching
        `eval*_perplexity` is added. HF's single `learning_rate` is dropped,
        since the two optimizer families have different LRs.
        """
        logs.pop("learning_rate", None)

        if "loss" in logs:
            self._log_window_losses(logs)
            elapsed = time.time() - self.log_start_time
            if elapsed > 0:
                current_step = self.state.global_step
                steps_passed = current_step - self.log_start_steps
                tokens_passed = self.total_tokens - self.log_start_tokens
                if steps_passed > 0:
                    logs["iters"] = current_step
                    logs["tokens_per_sec"] = tokens_passed / elapsed
                    logs["iters_per_sec"] = steps_passed / elapsed
                    logs["iter_time"] = elapsed / steps_passed
                    logs["samples_per_sec"] =\
                        (tokens_passed / self.block_size) / elapsed
                    if hasattr(self._llm, "last_router_loss") and self._llm.last_router_loss:
                        logs["router_loss"] = self._llm.last_router_loss.item()
                    # Per-expert load / token count / router importance,
                    # averaged over the trunk's MoE layers. These come from
                    # the last forward on this rank only (not reduced).
                    if getattr(self._llm.config, "experts", 1) > 1:
                        avg_load = torch.stack([layer.ffn.last_load.float() for layer in self._llm.layers]).mean(0)
                        avg_tokens = torch.stack([layer.ffn.last_num_tokens.float() for layer in self._llm.layers]).mean(0)
                        avg_importance = torch.stack([layer.ffn.last_importance.float() for layer in self._llm.layers]).mean(0)
                        for i in range(avg_load.numel()):
                            logs[f"expert_{i}_load"] = avg_load[i].item()
                            logs[f"expert_{i}_tokens"] = avg_tokens[i].item()
                            logs[f"expert_{i}_importance"] = avg_importance[i].item()

            # All groups of one algorithm share a peak LR and the scheduler
            # multiplier, so any one group's LR represents the family.
            if self.optimizer is not None:
                muon_lrs = {
                    group["lr"]
                    for group in self.optimizer.param_groups
                    if group.get("algorithm") == "muon"
                }

                adamw_lrs = {
                    group["lr"]
                    for group in self.optimizer.param_groups
                    if group.get("algorithm") == "adamw"
                }

                if muon_lrs:
                    muon_lr = next(iter(muon_lrs))
                    logs["muon_lr"] = muon_lr.item() if type(muon_lr) == torch.Tensor else muon_lr

                if adamw_lrs:
                    adamw_lr = next(iter(adamw_lrs))
                    logs["adamw_lr"] = adamw_lr.item() if type(adamw_lr) == torch.Tensor else adamw_lr

            # Next-token perplexity: from lm_loss, since "loss" can include
            # MTP heads, router loss and smaller Matryoshka sizes.
            try:
                logs["perplexity"] = math.exp(logs.get("lm_loss", logs["loss"]))
            except OverflowError:
                logs["perplexity"] = float("inf")

            self.log_start_time = time.time()
            self.log_start_tokens = self.total_tokens
            self.log_start_steps = self.state.global_step

        # eval_loss and Matryoshka eval_f<size>_loss -> *_perplexity
        for key in [k for k in logs if k.startswith("eval")
                    and k.endswith("_loss")]:
            try:
                logs[key[:-len("loss")] + "perplexity"] = math.exp(logs[key])
            except OverflowError:
                logs[key[:-len("loss")] + "perplexity"] = float("inf")

        super().log(logs, *args, **kwargs)

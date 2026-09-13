import math
import os
import time

import torch
from muonium import Muon
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import Trainer

from .data_utils import HellaswagDataset, hellaswag_collate_fn
from .misc_utils import human_readable_numbers as hrn


class LLMTrainer(Trainer):
    def __init__(
            self,
            *args,
            block_size: int = 4096,
            warmup_iters: int = 1000,
            last_decay_iter: int = 50000,
            muon_lr: float = 0.02,
            muon_weight_decay: float = 0.01,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.block_size = block_size
        self.warmup_iters = warmup_iters
        self.last_decay_iter = last_decay_iter

        self.muon_lr = muon_lr
        self.muon_weight_decay = muon_weight_decay

        self.total_tokens = 0
        self.log_start_time = time.time()
        self.log_start_tokens = 0
        self.log_start_steps = 0

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer

        muon_2d_params = []
        muon_q_params = []
        muon_k_up_params = []
        muon_v_up_params = []
        muon_3d_params = []
        adamw_decay_params = []
        adamw_nodecay_params = []

        n_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.sa_head_size
        rope_dim = self.model.config.rope_size
        nope_dim = head_dim - rope_dim

        q_split_sizes = (head_dim,) * n_heads
        k_up_split_sizes = (nope_dim,) * n_heads
        v_up_split_sizes = (head_dim,) * n_heads

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            is_router = "router" in name.lower()
            is_mhc = "mhc" in name.lower()
            is_embedding = "emb" in name.lower()
            is_head = "head" in name.lower()
            is_moe = "gate_up_weight" in name.lower()\
                or "down_weight" in name.lower()

            if is_moe and p.ndim >= 3:
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
            sum_p = lambda params: sum(p.numel() for p in params)
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

        self.optimizer = Muon([{
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
            "weight_decay": 0.0,
        }, {
            "params": adamw_decay_params,
            "algorithm": "adamw",
            "lr": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
        }], lr=self.muon_lr, wd=self.muon_weight_decay,
        orthogonalization_strategy="newton_schulz", use_cautious_wd=False,
        adam_betas=(self.args.adam_beta1, self.args.adam_beta2),
        is_adamw=True, foreach=None)

        return self.optimizer

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:
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
                return 1e-6 + (1.0 - 1e-6) * progress
            if step < decay_end:
                decay_progress =\
                    (step - warmup_iters) / (decay_end - warmup_iters)
                cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                return 0.1 + 0.9 * cosine
            return 0.1

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return self.lr_scheduler

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        loss = super().training_step(model, inputs, *args, **kwargs)

        if "input_ids" in inputs:
            local_tokens = inputs["input_ids"].numel()
            world_size =\
                self.args.world_size if hasattr(self.args, "world_size") else 1
            self.total_tokens += local_tokens * world_size
        return loss

    @torch.no_grad()
    def evaluate_hellaswag(self) -> float:
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

            outputs = self.model(**batch, return_per_sample_loss=True) # type: ignore

            preds = outputs.loss.reshape(-1, 4).argmin(dim=1)

            local_correct += (preds == 0).sum().item()
            local_total += preds.numel()

        correct = torch.tensor([local_correct], device=self.accelerator.device)
        total = torch.tensor([local_total], device=self.accelerator.device)

        correct = self.accelerator.gather(correct).sum() # type: ignore
        total = self.accelerator.gather(total).sum() # type: ignore

        return (correct.float() / total.float()).item()

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        self.model.eval() # type: ignore
        hellaswag_acc = self.evaluate_hellaswag()
        self.model.train() # type: ignore

        metrics["eval_hellaswag_acc"] = hellaswag_acc
        self.log({"eval_hellaswag_acc": hellaswag_acc})

        return metrics

    def log(self, logs: dict, *args, **kwargs) -> None:
        logs.pop("learning_rate", None)

        if "loss" in logs:
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
                    if hasattr(self.model, "last_lm_loss"):
                        logs["lm_loss"] = self.model.last_lm_loss.item() # type: ignore
                    if hasattr(self.model, "last_router_loss") and self.model.last_router_loss: # type: ignore
                        logs["router_loss"] = self.model.last_router_loss.item() # type: ignore
                    if getattr(self.model.config, "experts", 1) > 1: # type: ignore
                        avg_load = torch.stack([layer.ffn.last_load.float() for layer in self.model.layers]).mean(0) # type: ignore
                        avg_tokens = torch.stack([layer.ffn.last_num_tokens.float() for layer in self.model.layers]).mean(0) # type: ignore
                        avg_importance = torch.stack([layer.ffn.last_importance.float() for layer in self.model.layers]).mean(0) # type: ignore
                        for i in range(avg_load.numel()):
                            logs[f"expert_{i}_load"] = avg_load[i].item()
                            logs[f"expert_{i}_tokens"] = avg_tokens[i].item()
                            logs[f"expert_{i}_importance"] = avg_importance[i].item()

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

            try:
                logs["perplexity"] = math.exp(logs["loss"])
            except OverflowError:
                logs["perplexity"] = float("inf")

            self.log_start_time = time.time()
            self.log_start_tokens = self.total_tokens
            self.log_start_steps = self.state.global_step

        if "eval_loss" in logs:
            try:
                logs["eval_perplexity"] = math.exp(logs["eval_loss"])
            except OverflowError:
                logs["eval_perplexity"] = float("inf")

        super().log(logs, *args, **kwargs)

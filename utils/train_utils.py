import os
import math
import time
import torch
from transformers import Trainer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from typing import Optional

from .optimizer_utils import MultiOptimizer, MultiScheduler
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
        if self.optimizer is None:
            muon_params = []
            adamw_decay_params = []
            adamw_nodecay_params = []
            
            for name, p in self.model.named_parameters(): # type: ignore
                if p.requires_grad:
                    if p.dim() >= 2 and not\
                            any(k in name for k in ["emb", "head"]):
                        muon_params.append(p)
                    else:
                        if p.dim() >= 2:
                            adamw_decay_params.append(p)
                        else:
                            adamw_nodecay_params.append(p)

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                sum_p = lambda params: sum(p.numel() for p in params)
                print(f"Muon (2D Hidden): {len(muon_params)} tensors,"
                      f" {hrn(sum_p(muon_params))} params")
                print(f"AdamW (Decay): {len(adamw_decay_params)} tensors,"
                      f" {hrn(sum_p(adamw_decay_params))} params")
                print(f"AdamW (No Decay): {len(adamw_nodecay_params)} tensors,"
                      f" {hrn(sum_p(adamw_nodecay_params))} params")
            
            opt_muon = torch.optim.Muon(
                muon_params, lr=self.muon_lr,
                weight_decay=self.muon_weight_decay)

            use_fused = torch.cuda.is_available()\
                            and hasattr(torch.optim.AdamW, "_fused")
            adamw_groups = [{
                "params": adamw_nodecay_params,
                "weight_decay": 0.0,
            }, {
                "params": adamw_decay_params, 
                "weight_decay": self.args.weight_decay,
            }]
            opt_adamw = torch.optim.AdamW(
                adamw_groups, 
                lr=self.args.learning_rate, 
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon, 
                fused=use_fused
            )
            
            self.optimizer = MultiOptimizer([opt_muon, opt_adamw])
            
        return self.optimizer
    
    def create_scheduler(
            self,
            num_training_steps: int,
            optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        if self.lr_scheduler is None:
            opt_multi = optimizer if optimizer is not None else self.optimizer
            base_opt = opt_multi
            while hasattr(base_opt, "optimizer") and\
                    not hasattr(base_opt, "optimizers"):
                base_opt = base_opt.optimizer # type: ignore
            opt_muon, opt_adamw = base_opt.optimizers # type: ignore
            
            muon_warmup = LinearLR(
                opt_muon, start_factor=1e-6, total_iters=self.warmup_iters)
            muon_cosine = CosineAnnealingLR(
                opt_muon, eta_min=self.muon_lr / 10,
                T_max=self.last_decay_iter - self.warmup_iters)
            muon_const = LinearLR(
                opt_muon, start_factor=0.1, end_factor=0.1,
                total_iters=num_training_steps - self.last_decay_iter)
            muon_seq = SequentialLR(
                opt_muon, milestones=[self.warmup_iters, self.last_decay_iter],
                schedulers=[muon_warmup, muon_cosine, muon_const])

            adamw_warmup = LinearLR(
                opt_adamw, start_factor=1e-6, total_iters=self.warmup_iters)
            adamw_cosine = CosineAnnealingLR(
                opt_adamw, eta_min=self.args.learning_rate / 10,
                T_max=self.last_decay_iter - self.warmup_iters)
            adamw_const = LinearLR(
                opt_adamw, start_factor=0.1, end_factor=0.1,
                total_iters=num_training_steps - self.last_decay_iter)
            adamw_seq = SequentialLR(
                opt_adamw, milestones=[self.warmup_iters, self.last_decay_iter],
                schedulers=[adamw_warmup, adamw_cosine, adamw_const])
            self.lr_scheduler = MultiScheduler([muon_seq, adamw_seq], opt_multi)
            
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
            
            if self.optimizer is not None:
                base_opt = self.optimizer
                while hasattr(base_opt, "optimizer")\
                        and not hasattr(base_opt, "optimizers"):
                    base_opt = base_opt.optimizer # type: ignore
                    
                if hasattr(base_opt, "optimizers"):
                    logs["muon_lr"] =\
                        base_opt.optimizers[0].param_groups[0]["lr"]
                    logs["adamw_lr"] =\
                        base_opt.optimizers[1].param_groups[0]["lr"]
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

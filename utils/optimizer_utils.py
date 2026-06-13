import os
import torch
import argparse

from utils.misc_utils import human_readable_numbers


def get_optimizer(
        model: torch.nn.Module, args: argparse.Namespace
) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for p in model.parameters():
        if p.requires_grad:
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        sum_p = lambda params: sum(p.numel() for p in params)
        print(f"Decayed: {len(decay_params)} tensors, "
              f"{human_readable_numbers(sum_p(decay_params))} params")
        print(f"Non-decayed: {len(nodecay_params)} tensors, "
              f"{human_readable_numbers(sum_p(nodecay_params))} params")

    # 3. Check for fused AdamW availability (standard in modern PyTorch)
    use_fused = args.use_fused_optimizer and torch.cuda.is_available()
    use_fused = use_fused and hasattr(torch.optim.AdamW, "_fused")
    
    return torch.optim.AdamW(
        optim_groups, 
        lr=args.learning_rate, 
        betas=(args.betas[0], args.betas[1]),
        eps=args.optimizer_epsilon, 
        fused=use_fused
    )
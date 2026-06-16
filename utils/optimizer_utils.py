import torch
from typing import Optional

from utils.misc_utils import human_readable_numbers


class MultiOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)
        self.defaults = {}
        self.state = {} # type: ignore

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
            
    def state_dict(self):
        return {f"opt_{i}": opt.state_dict()\
                for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.optimizers):
            if f"opt_{i}" in state_dict:
                opt.load_state_dict(state_dict[f"opt_{i}"])


class MultiScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Wraps multiple schedulers to synchronize step() and get_last_lr()."""
    def __init__(
            self,
            schedulers: list[torch.optim.lr_scheduler.LRScheduler],
            optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        self.schedulers = schedulers
        self.optimizer = optimizer
        
    def step(self, *args, **kwargs):
        for sched in self.schedulers:
            if not args and not kwargs:
                sched.step()
            else:
                sched.step(*args, **kwargs)

    def get_last_lr(self):
        lrs = []
        for sched in self.schedulers:
            lrs.extend(sched.get_last_lr())
        return lrs
        
    def state_dict(self):
        return {f"sched_{i}": sched.state_dict()\
                for i, sched in enumerate(self.schedulers)}
        
    def load_state_dict(self, state_dict):
        for i, sched in enumerate(self.schedulers):
            if f"sched_{i}" in state_dict:
                sched.load_state_dict(state_dict[f"sched_{i}"])

import argparse
import torch, torch.nn as nn
from utils import get_batch


def train_step(
        model: nn.Module, optimizer: torch.optim.Optimizer,
        train_data: torch.Tensor, args: argparse.Namespace
) -> float:
    inputs, targets = get_batch(train_data, args.batch_size, args.block_size)
    inputs, targets = inputs.to(args.device), targets.to(args.device)
    _, loss = model(inputs, targets)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def evaluate(
        model: nn.Module, train_data: torch.Tensor,
        val_data: torch.Tensor, args: argparse.Namespace
) -> tuple[float, float]:
    def get_avg_loss(data: torch.Tensor) -> float:
        loss = 0
        for _ in range(args.eval_iters):
            loss += train_step(
                model, None, data, args)
        return loss / args.eval_iters

    model.eval()
    with torch.no_grad():
        avg_train_loss = get_avg_loss(train_data)
        avg_val_loss = get_avg_loss(val_data)
    model.train()
    return avg_train_loss, avg_val_loss
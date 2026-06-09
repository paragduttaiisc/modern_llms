import argparse, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Tuple, Optional, Any


def step(
        model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
        accelerator: Accelerator, optimizer: Optional[torch.optim.Optimizer]
) -> float:
    output = model(inputs, targets)
    loss = output.loss

    if optimizer is not None:
        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    reduced_loss = accelerator.reduce(loss, 'mean')
    return reduced_loss.item() # type: ignore


def evaluate(
    model: nn.Module, dataloader: DataLoader,
    accelerator: Accelerator, args: argparse.Namespace
) -> Tuple[float, float]:
    model.eval()
    total_validation_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for val_inputs, val_targets in dataloader:
            if val_steps >= args.eval_iters:
                break
            total_validation_loss += step(
                model, val_inputs, val_targets, accelerator, None)
            val_steps += 1
    model.train()
    avg_val_loss = total_validation_loss / args.eval_iters
    val_perplexity = math.exp(avg_val_loss)
    return avg_val_loss, val_perplexity


def train(
    model: nn.Module, optimizer: torch.optim.Optimizer,
    train_loader: DataLoader, val_loader: DataLoader,
    accelerator: Accelerator, args: argparse.Namespace,
    wandb_run: Optional[Any] = None,
) -> None:
    model.train()
    count = 0
    done = False
    total_train_loss = 0.0
    while not done:
        for inputs, targets in train_loader:
            if count > args.n_iters: # exit condition for training loop
                done = True
                break

            # Perform a training step and accumulate the loss for logging
            loss = step(model, inputs, targets, accelerator, optimizer)
            total_train_loss += loss

            if count % args.log_interval == 0: # log training loss
                accelerator.print(f"Step: {count}, Train loss: {loss:.4f}")
                if wandb_run is not None and accelerator.is_main_process:
                    wandb_run.log({
                        "train_loss": loss,
                        "lr": optimizer.param_groups[0]["lr"]
                    }, step=count)

            if count % args.eval_interval == 0: # evaluate on validation set
                avg_train_loss = total_train_loss / args.eval_interval
                if count == 0: avg_train_loss = loss
                total_train_loss = 0.0
                avg_val_loss, val_perplexity =\
                    evaluate(model, val_loader, accelerator, args)
                accelerator.print(
                    f"Step: {count},"
                    f" Avg Train loss: {torch.tensor(avg_train_loss):.4f},"
                    f" Avg Val loss: {torch.tensor(avg_val_loss):.4f},"
                    f" Avg Val Perplexity: {val_perplexity:.4f}"
                )
                if wandb_run is not None and accelerator.is_main_process:
                    wandb_run.log({
                        "avg_train_loss": avg_train_loss,
                        "avg_val_loss": avg_val_loss,
                        "avg_val_perplexity": val_perplexity,
                    }, step=count)
            if count % args.save_interval == 0 and count > 0: # save model
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.save_dir, safe_serialization=True)

            count += 1
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.save_dir, safe_serialization=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()

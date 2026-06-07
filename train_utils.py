import torch
from utils import get_batch


def train_step(model, optimizer, train_data, batch_size, block_size, device):
    inputs, targets = get_batch(train_data, batch_size, block_size)
    inputs, targets = inputs.to(device), targets.to(device)
    _, loss = model(inputs, targets)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def evaluate(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    def get_avg_loss(data):
        loss = 0
        for _ in range(eval_iters):
            loss += train_step(
                model, None, data, batch_size, block_size, device)
        return loss / eval_iters

    model.eval()
    with torch.no_grad():
        avg_train_loss = get_avg_loss(train_data)
        avg_val_loss = get_avg_loss(val_data)
    model.train()
    return avg_train_loss, avg_val_loss
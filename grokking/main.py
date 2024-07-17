import argparse
from math import ceil

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from data import get_dataloaders
from model import Transformer


def train_epoch(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> None:
    model.train()

    criterion = nn.CrossEntropyLoss()

    accs = []
    losses = []

    for batch in train_dataloader:

        batch = tuple(t.to(device) for t in batch)

        inputs, labels = batch

        optimizer.zero_grad()

        output = model(inputs)[-1, :, :]

        loss = criterion(output, labels)
        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

        loss.backward()

        optimizer.step()
        scheduler.step()

        accs.append(acc.item())
        losses.append(loss.item())

    metrics = {
        "accuracy": np.array(accs).mean(),
        "loss": np.array(losses).mean(),
    }
    return metrics


@torch.no_grad
def eval_epoch(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
) -> None:
    model.eval()

    criterion = nn.CrossEntropyLoss()

    correct = 0
    loss = 0.0

    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)

        inputs, labels = batch

        output = model(inputs)[-1, :, :]
        correct += (torch.argmax(output, dim=1) == labels).sum()
        loss += criterion(output, labels) * len(labels)

    acc = correct / len(val_dataloader.dataset)
    loss /= len(val_dataloader.dataset)

    metrics = {
        "accuracy": acc,
        "loss": loss,
    }
    return metrics


def main(config: argparse.Namespace) -> None:

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        config.operation,
        config.p,
        config.training_size,
        config.batch_size,
    )

    # Get model
    num_tokens = config.p + 2
    seq_len = 5

    if config.device == "cpu":
        device = torch.device("cpu")
    elif config.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("CUDA not available, using CPU")
            device = torch.device("cpu")
    elif config.device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("MPS not available, using CPU")
            device = torch.device("cpu")

    print(f"Using device: {device}")

    model = Transformer(
        config.num_layers,
        config.dim_model,
        config.num_heads,
        num_tokens,
        seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.lr_start_factor,
        total_iters=config.lr_total_iters,
    )

    num_epochs = ceil(config.n_steps / len(train_loader))

    train_losses = np.zeros(num_epochs)
    train_accs = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    val_accs = np.zeros(num_epochs)

    for epoch in (pbar := tqdm(range(num_epochs))):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)
        eval_metrics = eval_epoch(model, val_loader, device)

        train_losses[epoch] = train_metrics["loss"]
        train_accs[epoch] = train_metrics["accuracy"]
        val_losses[epoch] = eval_metrics["loss"]
        val_accs[epoch] = eval_metrics["accuracy"]

        pbar.set_description(
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Val Loss: {eval_metrics['loss']:.4f}, Val Acc: {eval_metrics['accuracy']:.4f}"
        )

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("results.png")
    if config.show_plots:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset
    dataset_parser = parser.add_argument_group("Dataset")
    dataset_parser.add_argument("--operation", type=str, default="addition")
    dataset_parser.add_argument("--p", type=int, default=97)
    dataset_parser.add_argument("--training_size", type=float, default=0.5)
    dataset_parser.add_argument("--batch_size", type=int, default=512)

    # Model
    model_parser = parser.add_argument_group("Model")
    model_parser.add_argument("--num_layers", type=int, default=2)
    model_parser.add_argument("--dim_model", type=int, default=128)
    model_parser.add_argument("--num_heads", type=int, default=4)
    model_parser.add_argument("--device", type=str, default="cpu")

    # Training
    training_parser = parser.add_argument_group("Training")
    training_parser.add_argument("--n_steps", type=int, default=int(1e5))
    training_parser.add_argument("--lr", type=float, default=1e-3)
    training_parser.add_argument("--lr_start_factor", type=float, default=0.1)
    training_parser.add_argument("--lr_total_iters", type=int, default=9)
    training_parser.add_argument("--weight_decay", type=float, default=1)

    parser.add_argument("--show_plots", action="store_true")

    args = parser.parse_args()

    main(args)

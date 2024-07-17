import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)

operations = {
    "addition": lambda x, y, p: (x + y) % p,
    "subtraction": lambda x, y, p: (x - y) % p,
    "division": lambda x, y, p: (x / y) % p,
}


def get_data(operation: str, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dataset consisting of z = (x op y) % p
    """
    op_token = p
    eq_token = p + 1

    x = torch.arange(0, p)
    y = torch.arange(0, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    labels = operations[operation](x, y, p)

    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels


def get_dataloaders(
    operation: str,
    p: int,
    training_size: float,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    inputs, labels = get_data(operation, p)
    dataset = TensorDataset(inputs, labels)

    train_size = int(training_size * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

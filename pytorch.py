"""PyTorch: Wine MLP and FashionMNIST prototype.

This module keeps the notebook logic in script form so the project can evolve
without changing the core data, model, training, or evaluation flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

try:
    from sklearn.datasets import load_wine
except ImportError as error:
    load_wine = None
    sklearn_import_error = error
else:
    sklearn_import_error = None


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    dropout: float = 0.2
    validation_split: float = 0.2
    seed: int = 42


CONFIG = TrainConfig()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path('.').resolve()
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts' / 'pytorch'
CHECKPOINT_DIR = ARTIFACTS_DIR / 'checkpoints'
TENSORBOARD_DIR = ARTIFACTS_DIR / 'tensorboard'
for directory in (ARTIFACTS_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG.seed)


class WineDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def load_wine_arrays() -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if load_wine is None:
        raise ImportError(
            'Missing scikit-learn. Run %pip install scikit-learn before using the Wine dataset.'
        ) from sklearn_import_error

    wine = load_wine()
    features = wine.data.astype(np.float32)
    labels = wine.target.astype(np.int64)
    metadata = {
        'feature_names': list(wine.feature_names),
        'target_names': list(wine.target_names),
        'num_samples': int(features.shape[0]),
        'num_features': int(features.shape[1]),
        'num_classes': int(len(np.unique(labels))),
    }
    return features, labels, metadata


def standardize(train_features: np.ndarray, valid_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-8
    return (train_features - mean) / std, (valid_features - mean) / std


def split_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    validation_split: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    split_index = int(len(features) * (1 - validation_split))
    train_indices = indices[:split_index]
    valid_indices = indices[split_index:]
    return (
        features[train_indices],
        features[valid_indices],
        labels[train_indices],
        labels[valid_indices],
    )


def build_wine_loaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader, Dict[str, object]]:
    features, labels, metadata = load_wine_arrays()
    train_features, valid_features, train_labels, valid_labels = split_arrays(
        features, labels, config.validation_split, config.seed
    )
    train_features, valid_features = standardize(train_features, valid_features)

    train_dataset = WineDataset(train_features, train_labels)
    valid_dataset = WineDataset(valid_features, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, valid_loader, metadata


class WineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_accuracy = 0.0
    total_examples = 0

    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        if is_training:
            optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_accuracy += accuracy_from_logits(logits, targets) * batch_size
        total_examples += batch_size

    return {
        'loss': total_loss / total_examples,
        'accuracy': total_accuracy / total_examples,
    }


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: TrainConfig,
) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: Dict[str, List[float]] = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
    }

    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        valid_metrics = run_epoch(model, valid_loader, criterion, None)

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['valid_loss'].append(valid_metrics['loss'])
        history['valid_acc'].append(valid_metrics['accuracy'])

        print(
            f'Epoch {epoch + 1:02d}/{config.epochs} | '
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} valid_acc={valid_metrics['accuracy']:.4f}"
        )

    return history


def plot_history(history: Dict[str, List[float]], title: str) -> None:
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], label='train')
    axes[0].plot(epochs, history['valid_loss'], label='valid')
    axes[0].set_title(f'{title} - loss')
    axes[0].set_xlabel('epoch')
    axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], label='train')
    axes[1].plot(epochs, history['valid_acc'], label='valid')
    axes[1].set_title(f'{title} - accuracy')
    axes[1].set_xlabel('epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


class FashionMNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_fashion_loaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(
        root=PROJECT_ROOT / 'data',
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=PROJECT_ROOT / 'data',
        train=False,
        download=True,
        transform=transform,
    )

    train_size = int(len(train_dataset) * 0.9)
    valid_size = len(train_dataset) - train_size
    train_subset, valid_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(CONFIG.seed),
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def train_with_checkpoint_and_tensorboard(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    experiment_name: str,
) -> Dict[str, List[float]]:
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR / experiment_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_valid_loss = float('inf')
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
    }

    for epoch in range(CONFIG.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        valid_metrics = run_epoch(model, valid_loader, criterion, None)

        for key, value in [
            ('train_loss', train_metrics['loss']),
            ('train_acc', train_metrics['accuracy']),
            ('valid_loss', valid_metrics['loss']),
            ('valid_acc', valid_metrics['accuracy']),
        ]:
            history[key].append(value)

        writer.add_scalar('loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('loss/valid', valid_metrics['loss'], epoch)
        writer.add_scalar('accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('accuracy/valid', valid_metrics['accuracy'], epoch)

        if valid_metrics['loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['loss']
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'valid_loss': best_valid_loss,
                },
                CHECKPOINT_DIR / f'{experiment_name}.pt',
            )

        print(
            f'Epoch {epoch + 1:02d}/{CONFIG.epochs} | '
            f"train_acc={train_metrics['accuracy']:.4f} valid_acc={valid_metrics['accuracy']:.4f}"
        )

    writer.close()
    return history


def predict_loader(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(DEVICE))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets_list.append(targets.numpy())
    return np.concatenate(predictions), np.concatenate(targets_list)


def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for target, prediction in zip(y_true, y_pred):
        matrix[target, prediction] += 1
    return matrix


def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray) -> List[Tuple[int, int]]:
    mistakes = np.where(y_true != y_pred)[0]
    return [(int(index), int(y_true[index])) for index in mistakes]


PROJECT_TREE = """
src/
    data/
    models/
    training/
    evaluation/
config.yaml
notebooks/
    pytorch.ipynb
    tensorflow.ipynb
"""


def main() -> None:
    print(f'Using device: {DEVICE}')
    print('Artifacts directory:', ARTIFACTS_DIR)
    print('Suggested modular structure:')
    print(PROJECT_TREE)

    train_loader, valid_loader, wine_metadata = build_wine_loaders(CONFIG)
    wine_model = WineMLP(
        input_dim=wine_metadata['num_features'],
        hidden_dim=CONFIG.hidden_dim,
        num_classes=wine_metadata['num_classes'],
        dropout=CONFIG.dropout,
    ).to(DEVICE)

    # history = fit_classifier(wine_model, train_loader, valid_loader, CONFIG)
    # plot_history(history, 'Wine MLP')

    fashion_train_loader, fashion_valid_loader, fashion_test_loader = build_fashion_loaders(CONFIG.batch_size)
    fashion_model = FashionMNISTCNN().to(DEVICE)

    # fashion_history = train_with_checkpoint_and_tensorboard(
    #     fashion_model,
    #     fashion_train_loader,
    #     fashion_valid_loader,
    #     'fashion_cnn',
    # )
    # y_pred, y_true = predict_loader(fashion_model, fashion_test_loader)
    # cm = confusion_matrix_numpy(y_true, y_pred, num_classes=10)
    # mistakes = analyze_errors(y_true, y_pred)


if __name__ == '__main__':
    main()

"""Entrena el detector perro vs no-perro con una CNN **desde cero** en PyTorch.

Réplica del modelo de `train.py` (TensorFlow) para comparar frameworks sobre el
MISMO dataset: misma arquitectura (4 bloques conv + GlobalAveragePooling), data
augmentation, early stopping y LR scheduling. Usa GPU si está disponible.

Uso:
    python train_torch.py                # 30 epochs
    python train_torch.py --epochs 40
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import dataset as dataset_module


@dataclass(frozen=True)
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    dropout: float = 0.4
    validation_split: float = 0.2
    patience: int = 10
    seed: int = 42
    num_workers: int = 4


CONFIG = Config()
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = dataset_module.RAW_ROOT
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'dog_detector_torch.pt'
HISTORY_PLOT = ARTIFACTS_DIR / 'dog_detector_torch_history.png'
CONFUSION_MATRIX_PLOT = ARTIFACTS_DIR / 'dog_detector_torch_confusion_matrix.png'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(config: Config) -> Tuple[DataLoader, DataLoader, List[str]]:
    """ImageFolder + split 80/20 con seed fijo. ToTensor escala a [0,1] (como TF)."""
    size = config.image_size
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    # Dos vistas del mismo dataset (transforms distintos) y el mismo split de índices.
    full_train = datasets.ImageFolder(str(DATA_ROOT), transform=train_tf)
    full_eval = datasets.ImageFolder(str(DATA_ROOT), transform=eval_tf)
    n = len(full_train)
    generator = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(n, generator=generator).tolist()
    n_val = int(n * config.validation_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = Subset(full_train, train_idx)
    valid_ds = Subset(full_eval, val_idx)

    pin = DEVICE.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=pin)
    return train_loader, valid_loader, full_train.classes


class DogDetector(nn.Module):
    """Misma arquitectura que la CNN de TensorFlow: 4 bloques + GAP, salida 1 logit."""

    def __init__(self, dropout: float = 0.4) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int, n_convs: int, drop: float) -> nn.Sequential:
            layers: List[nn.Module] = []
            channels = in_ch
            for _ in range(n_convs):
                layers += [nn.Conv2d(channels, out_ch, 3, padding=1, bias=False),
                           nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
                channels = out_ch
            layers += [nn.MaxPool2d(2), nn.Dropout(drop)]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, 32, 1, 0.10),
            block(32, 64, 1, 0.15),
            block(64, 128, 2, 0.20),
            block(128, 256, 2, 0.30),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))  # logits (N, 1)


def run_epoch(model, loader, criterion, optimizer=None) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += ((torch.sigmoid(logits) >= 0.5).float() == targets).sum().item()
            total += images.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, class_names: List[str]) -> Dict[str, object]:
    model.eval()
    probs, trues = [], []
    for images, labels in loader:
        logits = model(images.to(DEVICE))
        probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
        trues.append(labels.numpy().ravel())
    prob = np.concatenate(probs)
    true = np.concatenate(trues).astype(int)
    pred = (prob >= 0.5).astype(int)

    matrix = np.zeros((2, 2), dtype=int)
    for t, p in zip(true, pred):
        matrix[t, p] += 1
    accuracy = float((pred == true).mean())
    auc = float(roc_auc_score(true, prob))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de confusión')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PLOT, dpi=120)
    plt.close(fig)

    print(f'\nClases: {class_names}')
    print('Matriz de confusión (filas=real, columnas=predicho):')
    print(matrix)
    print(f'Accuracy en validación: {accuracy:.4f} | AUC: {auc:.4f}')
    print('Matriz de confusión guardada en', CONFUSION_MATRIX_PLOT)
    return {'accuracy': accuracy, 'auc': auc, 'confusion_matrix': matrix, 'class_names': class_names}


def plot_history(history: Dict[str, List[float]], save_path: Path) -> None:
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history['train_loss'], label='train')
    axes[0].plot(epochs, history['val_loss'], label='valid')
    axes[0].set_title('loss'); axes[0].set_xlabel('epoch'); axes[0].legend()
    axes[1].plot(epochs, history['train_acc'], label='train')
    axes[1].plot(epochs, history['val_acc'], label='valid')
    axes[1].set_title('accuracy'); axes[1].set_xlabel('epoch'); axes[1].legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
    print('Gráfica guardada en', save_path)


def run(epochs: int = CONFIG.epochs) -> Dict[str, object]:
    print('PyTorch:', torch.__version__, '| device:', DEVICE)
    if not dataset_module.is_ready():
        raise SystemExit('No hay dataset. Corre primero: python dataset.py')

    set_seed(CONFIG.seed)
    train_loader, valid_loader, class_names = build_loaders(CONFIG)
    print('Clases:', class_names)

    model = DogDetector(CONFIG.dropout).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, None)
        scheduler.step(val_loss)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} '
              f'| val_loss={val_loss:.4f} acc={val_acc:.4f} | lr={lr:.2e}')

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG.patience:
                print(f'Early stopping en epoch {epoch} (mejor val_loss={best_val:.4f})')
                break

    model.load_state_dict(best_state)  # restaurar mejores pesos
    plot_history(history, HISTORY_PLOT)
    results = evaluate(model, valid_loader, class_names)
    print('\nModelo guardado en', MODEL_PATH)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Entrena el detector perro vs no-perro en PyTorch.')
    parser.add_argument('--epochs', type=int, default=CONFIG.epochs)
    args = parser.parse_args()
    run(epochs=args.epochs)


if __name__ == '__main__':
    main()

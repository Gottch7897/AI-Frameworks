"""Entrena el detector perro vs no-perro con una CNN **desde cero** en PyTorch.

Réplica del modelo de `train_tf.py` para comparar frameworks sobre el
MISMO dataset: misma arquitectura (4 bloques conv + GlobalAveragePooling), data
augmentation, early stopping y LR scheduling. Usa GPU si está disponible.

Uso:
    python train_torch.py                # 30 epochs
    python train_torch.py --epochs 40
    python train_torch.py --optuna-trials 8 --optuna-epochs 4
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
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
    optuna_trials: int = 0
    optuna_epochs: int = 4
    optuna_study_name: str = 'dog_detector_torch'


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filters: Tuple[int, ...]
    convs_per_block: Tuple[int, ...]
    block_dropouts: Tuple[float, ...]
    dense_units: int
    dropout: float
    imbalance: str


CONFIG = Config()
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = dataset_module.RAW_ROOT
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MODEL_SPECS: Dict[str, ModelSpec] = {
    'torch_small_sampler': ModelSpec(
        name='torch_small_sampler',
        filters=(32, 64, 128),
        convs_per_block=(1, 1, 2),
        block_dropouts=(0.10, 0.15, 0.20),
        dense_units=128,
        dropout=0.30,
        imbalance='weighted_sampler',
    ),
    'torch_medium_pos_weight': ModelSpec(
        name='torch_medium_pos_weight',
        filters=(32, 64, 128, 192),
        convs_per_block=(1, 1, 2, 2),
        block_dropouts=(0.10, 0.15, 0.20, 0.25),
        dense_units=256,
        dropout=0.40,
        imbalance='pos_weight',
    ),
    'torch_deep_focal': ModelSpec(
        name='torch_deep_focal',
        filters=(32, 64, 128, 256),
        convs_per_block=(1, 1, 2, 2),
        block_dropouts=(0.10, 0.15, 0.20, 0.30),
        dense_units=256,
        dropout=0.45,
        imbalance='focal_loss',
    ),
}


def artifact_paths(variant: str) -> Dict[str, Path]:
    prefix = f'dog_detector_torch_{variant}'
    return {
        'model': ARTIFACTS_DIR / f'{prefix}.pt',
        'history': ARTIFACTS_DIR / f'{prefix}_history.png',
        'confusion': ARTIFACTS_DIR / f'{prefix}_confusion_matrix.png',
        'metrics': ARTIFACTS_DIR / f'{prefix}_metrics.json',
        'optuna': ARTIFACTS_DIR / f'{prefix}_optuna_best_params.json',
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_splits(config: Config) -> Tuple[Subset, Subset, List[str], np.ndarray, np.ndarray]:
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

    full_train = datasets.ImageFolder(str(DATA_ROOT), transform=train_tf)
    full_eval = datasets.ImageFolder(str(DATA_ROOT), transform=eval_tf)
    n = len(full_train)
    generator = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(n, generator=generator).tolist()
    n_val = int(n * config.validation_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = Subset(full_train, train_idx)
    valid_ds = Subset(full_eval, val_idx)
    train_labels = np.asarray([full_train.targets[i] for i in train_idx], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=2)

    return train_ds, valid_ds, full_train.classes, train_labels, class_counts


def build_loaders(config: Config, spec: ModelSpec) -> Tuple[DataLoader, DataLoader, List[str], np.ndarray, np.ndarray]:
    train_ds, valid_ds, class_names, train_labels, class_counts = build_splits(config)

    pin = DEVICE.type == 'cuda'
    if spec.imbalance == 'weighted_sampler':
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0).astype(np.float64)
        sample_weights = torch.as_tensor(class_weights[train_labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler,
                                  num_workers=config.num_workers, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=pin)
    return train_loader, valid_loader, class_names, train_labels, class_counts


class DogDetector(nn.Module):
    """Misma arquitectura que la CNN de TensorFlow: 4 bloques + GAP, salida 1 logit."""

    def __init__(self, filters: Tuple[int, ...], convs_per_block: Tuple[int, ...], block_dropouts: Tuple[float, ...], dropout: float = 0.4, dense_units: int = 256) -> None:
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

        feature_blocks: List[nn.Module] = []
        channels = 3
        for out_channels, n_convs, block_drop in zip(filters, convs_per_block, block_dropouts):
            feature_blocks.append(block(channels, out_channels, n_convs, block_drop))
            channels = out_channels
        self.features = nn.Sequential(*feature_blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, dense_units), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt).pow(self.gamma) * bce
        return focal.mean()


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
def evaluate(model, loader, class_names: List[str], save_path: Path) -> Dict[str, object]:
    model.eval()
    probs, trues = [], []
    for images, labels in loader:
        logits = model(images.to(DEVICE))
        probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
        trues.append(labels.numpy().ravel())
    prob = np.concatenate(probs)
    true = np.concatenate(trues).astype(int)
    pred = (prob >= 0.5).astype(int)

    matrix = sk_confusion_matrix(true, pred, labels=[0, 1])
    accuracy = float(accuracy_score(true, pred))
    precision = float(precision_score(true, pred, zero_division=0))
    recall = float(recall_score(true, pred, zero_division=0))
    f1 = float(f1_score(true, pred, zero_division=0))
    auc = float(roc_auc_score(true, prob)) if len(np.unique(true)) > 1 else float('nan')

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
    fig.savefig(save_path, dpi=120)
    plt.close(fig)

    print(f'\nClases: {class_names}')
    print('Matriz de confusión (filas=real, columnas=predicho):')
    print(matrix)
    print(f'Accuracy en validación: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}')
    print('Matriz de confusión guardada en', save_path)
    print('Interpretación: recall mide cobertura de la clase positiva; precision mide cuántos positivos predichos son correctos.')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'confusion_matrix': matrix, 'class_names': class_names}


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


def train_model(config: Config, train_loader: DataLoader, valid_loader: DataLoader, spec: ModelSpec,
                hyperparams: Dict[str, object], epochs: int, class_counts: np.ndarray) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    set_seed(config.seed)
    model = DogDetector(
        filters=spec.filters,
        convs_per_block=spec.convs_per_block,
        block_dropouts=spec.block_dropouts,
        dropout=float(hyperparams['dropout']),
        dense_units=int(hyperparams['dense_units']),
    ).to(DEVICE)
    if spec.imbalance == 'pos_weight':
        pos_count = max(int(class_counts[1]), 1)
        neg_count = max(int(class_counts[0]), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_count / pos_count], dtype=torch.float32, device=DEVICE))
    elif spec.imbalance == 'focal_loss':
        criterion = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer_name = str(hyperparams['optimizer'])
    optimizer = {
        'adam': torch.optim.Adam(model.parameters(), lr=float(hyperparams['learning_rate'])),
        'sgd': torch.optim.SGD(model.parameters(), lr=float(hyperparams['learning_rate']), momentum=0.9),
        'rmsprop': torch.optim.RMSprop(model.parameters(), lr=float(hyperparams['learning_rate'])),
    }[optimizer_name]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, None)
        scheduler.step(val_loss)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= max(2, config.patience // 2):
                break
    model.load_state_dict(best_state)
    return model, history


def run_optuna(config: Config, train_loader: DataLoader, valid_loader: DataLoader, spec: ModelSpec, class_counts: np.ndarray) -> optuna.study.Study:
    print(f'Ejecutando Optuna para PyTorch ({spec.name})...')

    def objective(trial: optuna.Trial) -> float:
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.15, 0.5),
            'dense_units': trial.suggest_categorical('dense_units', [128, 256]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
        }
        _, history = train_model(config, train_loader, valid_loader, spec, hyperparams, config.optuna_epochs, class_counts)
        val_acc = history['val_acc']
        return float(max(val_acc)) if val_acc else 0.0

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name=config.optuna_study_name)
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)
    return study


def run_single_variant(
    variant: str,
    epochs: int = CONFIG.epochs,
    optuna_trials: int = CONFIG.optuna_trials,
    optuna_epochs: int = CONFIG.optuna_epochs,
) -> Dict[str, object]:
    spec = MODEL_SPECS[variant]
    paths = artifact_paths(variant)
    print(f'PyTorch: {torch.__version__} | device: {DEVICE} | variante: {variant}')
    if not dataset_module.is_ready():
        raise SystemExit('No hay dataset. Corre primero: python dataset.py')

    set_seed(CONFIG.seed)
    train_loader, valid_loader, class_names, _, class_counts = build_loaders(CONFIG, spec)
    print('Clases:', class_names)

    if optuna_trials > 0:
        config = Config(epochs=epochs, optuna_trials=optuna_trials, optuna_epochs=optuna_epochs)
        study = run_optuna(config, train_loader, valid_loader, spec, class_counts)
        print(f'\nMejores hiperparámetros encontrados con Optuna para {variant}:')
        for key, value in study.best_params.items():
            print(f'  {key}: {value}')
        with paths['optuna'].open('w', encoding='utf-8') as handle:
            json.dump(study.best_params, handle, indent=2)
        hyperparams = study.best_params
    else:
        hyperparams = {
            'learning_rate': CONFIG.learning_rate,
            'dropout': CONFIG.dropout,
            'dense_units': 256,
            'optimizer': 'adam',
        }

    model, history = train_model(CONFIG, train_loader, valid_loader, spec, hyperparams, epochs, class_counts)
    plot_history(history, paths['history'])
    results = evaluate(model, valid_loader, class_names, paths['confusion'])
    torch.save({'model_state_dict': model.state_dict(), 'variant': variant, 'spec': spec.__dict__, 'class_names': class_names}, paths['model'])
    with paths['metrics'].open('w', encoding='utf-8') as handle:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in results.items()}, handle, indent=2)
    print('\nModelo guardado en', paths['model'])
    return results


def run_all_models(epochs: int, optuna_trials: int, optuna_epochs: int) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for variant in MODEL_SPECS:
        print(f'\n=== Entrenando {variant} ===')
        results[variant] = run_single_variant(variant, epochs=epochs, optuna_trials=0 if len(MODEL_SPECS) > 1 else optuna_trials, optuna_epochs=optuna_epochs)
    return results


def run(
    variant: str = 'torch_small_sampler',
    epochs: int = CONFIG.epochs,
    optuna_trials: int = CONFIG.optuna_trials,
    optuna_epochs: int = CONFIG.optuna_epochs,
    all_models: bool = False,
) -> Dict[str, object]:
    if all_models:
        return run_all_models(epochs=epochs, optuna_trials=optuna_trials, optuna_epochs=optuna_epochs)
    return run_single_variant(variant, epochs=epochs, optuna_trials=optuna_trials, optuna_epochs=optuna_epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description='Entrena el detector perro vs no-perro en PyTorch.')
    parser.add_argument('--variant', type=str, default='torch_small_sampler', choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument('--epochs', type=int, default=CONFIG.epochs)
    parser.add_argument('--optuna-trials', type=int, default=CONFIG.optuna_trials)
    parser.add_argument('--optuna-epochs', type=int, default=CONFIG.optuna_epochs)
    parser.add_argument('--all-models', action='store_true', help='Entrenar y guardar las 3 variantes de PyTorch.')
    args = parser.parse_args()
    run(variant=args.variant, epochs=args.epochs, optuna_trials=args.optuna_trials, optuna_epochs=args.optuna_epochs, all_models=args.all_models)


if __name__ == '__main__':
    main()

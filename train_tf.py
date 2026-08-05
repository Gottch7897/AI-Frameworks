"""Entrena el detector perro vs no-perro con una CNN **desde cero** (sin preentrenar).

Lee las imágenes de `data/dog_vs_notdog/raw/{dog,not_dog}` y entrena una CNN con
data augmentation, `class_weight`, early stopping y LR scheduling. Evalúa sobre la
validación, guarda el modelo y una gráfica de curvas.

Usa GPU automáticamente si TensorFlow la detecta; si no, corre en CPU (más lento).

Uso:
    python train_tf.py                 # entrena 30 epochs
    python train_tf.py --epochs 40
    python train_tf.py --optuna-trials 8 --optuna-epochs 4
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use('Agg')  # backend sin ventana: guarda la gráfica a archivo
import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import dataset as dataset_module


@dataclass(frozen=True)
class Config: #JUMP1
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    dropout: float = 0.4
    validation_split: float = 0.2
    patience: int = 10
    seed: int = 42
    optuna_trials: int = 0
    optuna_epochs: int = 4
    optuna_study_name: str = 'dog_detector_tf'


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filters: Tuple[int, ...]
    convs_per_block: Tuple[int, ...]
    block_dropouts: Tuple[float, ...]
    dense_units: int
    dropout: float
    imbalance: str
    augment: bool = True


CONFIG = Config()
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = dataset_module.RAW_ROOT
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
TENSORBOARD_DIR = ARTIFACTS_DIR / 'tensorboard'
for directory in (ARTIFACTS_DIR, TENSORBOARD_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
tf.get_logger().setLevel('ERROR')
tf.keras.utils.set_random_seed(CONFIG.seed)


MODEL_SPECS: Dict[str, ModelSpec] = {
    'tf_small_class_weight': ModelSpec(
        name='tf_small_class_weight',
        filters=(32, 64, 128),
        convs_per_block=(1, 1, 2),
        block_dropouts=(0.10, 0.15, 0.20),
        dense_units=128,
        dropout=0.30,
        imbalance='class_weight',
    ),
    'tf_medium_focal': ModelSpec(
        name='tf_medium_focal',
        filters=(32, 64, 128, 192),
        convs_per_block=(1, 1, 2, 2),
        block_dropouts=(0.10, 0.15, 0.20, 0.25),
        dense_units=256,
        dropout=0.40,
        imbalance='focal_loss',
    ),
    'tf_deep_oversample': ModelSpec(
        name='tf_deep_oversample',
        filters=(32, 64, 128, 256),
        convs_per_block=(1, 1, 2, 2),
        block_dropouts=(0.10, 0.15, 0.20, 0.30),
        dense_units=256,
        dropout=0.45,
        imbalance='oversample',
    ),
}


def artifact_paths(variant: str) -> Dict[str, Path]:
    prefix = f'dog_detector_tf_{variant}'
    return {
        'model': ARTIFACTS_DIR / f'{prefix}.keras',
        'history': ARTIFACTS_DIR / f'{prefix}_history.png',
        'confusion': ARTIFACTS_DIR / f'{prefix}_confusion_matrix.png',
        'metrics': ARTIFACTS_DIR / f'{prefix}_metrics.json',
        'optuna': ARTIFACTS_DIR / f'{prefix}_optuna_best_params.json',
    }


def build_datasets(config: Config) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], Dict[str, int]]:
    """Carga imágenes desde carpetas y arma train/val (una subcarpeta por clase)."""
    common = dict(
        directory=str(DATA_ROOT),
        labels='inferred',
        label_mode='binary',
        image_size=config.image_size,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        seed=config.seed,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset='training', shuffle=True, **common)
    valid_ds = tf.keras.utils.image_dataset_from_directory(subset='validation', shuffle=True, **common)
    class_names = train_ds.class_names
    class_counts = dataset_module.counts()
    return train_ds, valid_ds, class_names, class_counts


def build_cnn(
    image_size: Tuple[int, int],
    dropout: float,
    dense_units: int = 256,
    augment: bool = True,
    filters: Tuple[int, ...] = (32, 64, 128, 256),
    convs_per_block: Tuple[int, ...] = (1, 1, 2, 2),
    block_dropouts: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.30),
) -> tf.keras.Model: #JUMP2
    """CNN desde cero con bloques configurables, augmentation y cabeza densa."""
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = inputs
    if augment:
        x = tf.keras.layers.RandomFlip('horizontal')(x)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
        x = tf.keras.layers.RandomContrast(0.1)(x)

    def conv_block(tensor, block_filters: int, n_convs: int, drop: float):
        for _ in range(n_convs):
            tensor = tf.keras.layers.Conv2D(block_filters, 3, padding='same', use_bias=False)(tensor)
            tensor = tf.keras.layers.BatchNormalization()(tensor)
            tensor = tf.keras.layers.ReLU()(tensor)
        tensor = tf.keras.layers.MaxPooling2D()(tensor)
        return tf.keras.layers.Dropout(drop)(tensor)

    for block_filters, n_convs, block_drop in zip(filters, convs_per_block, block_dropouts):
        x = conv_block(x, block_filters, n_convs, block_drop)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='dog_detector')


def normalize_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    autotune = tf.data.AUTOTUNE
    return dataset.map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune).prefetch(autotune)


def build_oversampled_train_dataset(train_ds: tf.data.Dataset, batch_size: int, seed: int) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE
    unbatched = train_ds.unbatch()
    class_zero = unbatched.filter(lambda x, y: tf.equal(y, 0)).shuffle(1000, seed=seed, reshuffle_each_iteration=True).repeat()
    class_one = unbatched.filter(lambda x, y: tf.equal(y, 1)).shuffle(1000, seed=seed, reshuffle_each_iteration=True).repeat()
    balanced = tf.data.Dataset.sample_from_datasets([class_zero, class_one], weights=[0.5, 0.5], seed=seed)
    balanced = balanced.batch(batch_size)
    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    return balanced.map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune).prefetch(autotune)


def class_weight_from_dirs(class_names: List[str]) -> Dict[int, float]:
    counts = [len(dataset_module._list_images(DATA_ROOT / name)) for name in class_names]
    total = sum(counts) or 1
    return {i: total / (len(counts) * c) for i, c in enumerate(counts) if c > 0}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for target, prediction in zip(y_true, y_pred):
        matrix[int(target), int(prediction)] += 1
    return matrix


def get_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    opts = {
        'adam': tf.keras.optimizers.Adam(learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate, momentum=0.9),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate),
    }
    return opts[name]


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset, class_names: List[str], save_path: Path) -> Dict[str, object]:
    prob_batches, true_batches = [], []
    for features, labels in dataset:
        prob_batches.append(model.predict(features, verbose=0).ravel())
        true_batches.append(labels.numpy().ravel())
    probabilities = np.concatenate(prob_batches)
    predicted = (probabilities >= 0.5).astype(int)
    true = np.concatenate(true_batches).astype(int)

    matrix = sk_confusion_matrix(true, predicted, labels=[0, 1])
    accuracy = float(accuracy_score(true, predicted))
    precision = float(precision_score(true, predicted, zero_division=0))
    recall = float(recall_score(true, predicted, zero_division=0))
    f1 = float(f1_score(true, predicted, zero_division=0))
    auc = float(roc_auc_score(true, probabilities)) if len(np.unique(true)) > 1 else float('nan')

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
    print('Interpretación: recall mide cuánto detecta la clase positiva; precision mide cuántos positivos predichos son correctos.')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': matrix,
        'class_names': class_names,
    }


def plot_history(history: tf.keras.callbacks.History, save_path: Path) -> None:
    metrics = history.history
    epochs = np.arange(1, len(metrics['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, metrics['loss'], label='train')
    axes[0].plot(epochs, metrics['val_loss'], label='valid')
    axes[0].set_title('loss'); axes[0].set_xlabel('epoch'); axes[0].legend()
    axes[1].plot(epochs, metrics['accuracy'], label='train')
    axes[1].plot(epochs, metrics['val_accuracy'], label='valid')
    axes[1].set_title('accuracy'); axes[1].set_xlabel('epoch'); axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print('Gráfica guardada en', save_path)


def prepare_datasets_for_spec(
    config: Config,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    spec: ModelSpec,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[int, float] | None, tf.keras.losses.Loss | str]:
    train_ds_standard = normalize_dataset(train_ds)
    valid_ds_standard = normalize_dataset(valid_ds)
    if spec.imbalance == 'class_weight':
        return train_ds_standard, valid_ds_standard, class_weight_from_dirs(['dog', 'not_dog']), 'binary_crossentropy'
    if spec.imbalance == 'focal_loss':
        return train_ds_standard, valid_ds_standard, None, FocalLossTF(gamma=2.0, alpha=0.25)
    if spec.imbalance == 'oversample':
        return build_oversampled_train_dataset(train_ds, config.batch_size, config.seed), valid_ds_standard, None, 'binary_crossentropy'
    raise ValueError(f'Estrategia de desbalance no soportada: {spec.imbalance}')


def train_model(
    config: Config,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    spec: ModelSpec,
    hyperparams: Dict[str, object],
    epochs: int,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    model = build_cnn(
        config.image_size,
        float(hyperparams['dropout']),
        dense_units=int(hyperparams['dense_units']),
        augment=bool(hyperparams['augment']),
        filters=spec.filters,
        convs_per_block=spec.convs_per_block,
        block_dropouts=spec.block_dropouts,
    )
    prepared_train_ds, prepared_valid_ds, class_weight, loss = prepare_datasets_for_spec(config, train_ds, valid_ds, spec)
    model.compile(
        optimizer=get_optimizer(str(hyperparams['optimizer']), float(hyperparams['learning_rate'])),
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=max(2, config.patience // 2), restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1, config.patience // 4), min_lr=1e-5),
    ]
    history = model.fit(
        prepared_train_ds,
        validation_data=prepared_valid_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )
    return model, history


def run_optuna(config: Config, train_ds: tf.data.Dataset, valid_ds: tf.data.Dataset, spec: ModelSpec) -> optuna.study.Study: #JUMP4, probar combinaciones
    print(f'Ejecutando Optuna para TensorFlow ({spec.name})...')

    def objective(trial: optuna.Trial) -> float:
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.15, 0.5),
            'dense_units': trial.suggest_categorical('dense_units', [64, 128, 256]),
            'augment': trial.suggest_categorical('augment', [True, False]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
        }
        _, history = train_model(config, train_ds, valid_ds, spec, hyperparams, config.optuna_epochs)
        val_acc = history.history.get('val_accuracy', [])
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
    """Entrena, evalúa y guarda un modelo concreto. Devuelve las métricas finales."""
    spec = MODEL_SPECS[variant]
    paths = artifact_paths(variant)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow: {tf.__version__} | GPU: {'sí' if gpus else 'no (CPU)'} | variante: {variant}")

    if not dataset_module.is_ready():
        raise SystemExit('No hay dataset. Corre primero: python dataset.py')

    train_ds, valid_ds, class_names, _ = build_datasets(CONFIG)

    if optuna_trials > 0:
        config = Config(epochs=epochs, optuna_trials=optuna_trials, optuna_epochs=optuna_epochs)
        study = run_optuna(config, train_ds, valid_ds, spec)
        best_params = study.best_params
        print(f'\nMejores hiperparámetros encontrados con Optuna para {variant}:') #JUMP5
        for key, value in best_params.items():
            print(f'  {key}: {value}')
        with paths['optuna'].open('w', encoding='utf-8') as handle:
            json.dump(best_params, handle, indent=2) #JUMP6
        hyperparams = best_params
    else:
        hyperparams = {
            'learning_rate': CONFIG.learning_rate,
            'dropout': CONFIG.dropout,
            'dense_units': 256,
            'augment': True,
            'optimizer': 'adam',
        }

    model, history = train_model(CONFIG, train_ds, valid_ds, spec, hyperparams, epochs)
    plot_history(history, paths['history'])
    results = evaluate(model, normalize_dataset(valid_ds), class_names, paths['confusion'])
    model.save(paths['model'])
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
    variant: str = 'tf_small_class_weight',
    epochs: int = CONFIG.epochs,
    optuna_trials: int = CONFIG.optuna_trials,
    optuna_epochs: int = CONFIG.optuna_epochs,
    all_models: bool = False,
) -> Dict[str, object]:
    if all_models:
        return run_all_models(epochs=epochs, optuna_trials=optuna_trials, optuna_epochs=optuna_epochs)
    return run_single_variant(variant, epochs=epochs, optuna_trials=optuna_trials, optuna_epochs=optuna_epochs)


def main() -> None:
    parser = argparse.ArgumentParser(description='Entrena el detector perro vs no-perro (desde cero).')
    parser.add_argument('--variant', type=str, default='tf_small_class_weight', choices=sorted(MODEL_SPECS.keys()), help='Variante de modelo a entrenar.')
    parser.add_argument('--epochs', type=int, default=CONFIG.epochs, help='Número de epochs.')
    parser.add_argument('--optuna-trials', type=int, default=CONFIG.optuna_trials, help='Número de trials de Optuna a ejecutar.')
    parser.add_argument('--optuna-epochs', type=int, default=CONFIG.optuna_epochs, help='Epochs por trial de Optuna.')
    parser.add_argument('--all-models', action='store_true', help='Entrenar y guardar las 3 variantes de TensorFlow.')
    args = parser.parse_args()
    run(variant=args.variant, epochs=args.epochs, optuna_trials=args.optuna_trials, optuna_epochs=args.optuna_epochs, all_models=args.all_models)


if __name__ == '__main__':
    main()

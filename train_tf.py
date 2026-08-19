"""EcoSort — entrena un detector binario de material (TensorFlow, desde cero).

Un modelo por material: `--target {plastico,vidrio,papel}`. Misma CNN desde cero
(4 bloques + GlobalAveragePooling), data augmentation, `class_weight` (clave para
el modelo desequilibrado 'vidrio'), early stopping y LR scheduling. Opcional:
búsqueda de hiperparámetros con Optuna.

Uso:
    python train_tf.py --target plastico --epochs 25
    python train_tf.py --target vidrio --epochs 25 --optuna-trials 10 --optuna-epochs 8
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import dataset as dataset_module

try:
    import optuna  # solo para --optuna-trials
except ImportError:
    optuna = None


@dataclass(frozen=True)
class Config:
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 1e-3
    dropout: float = 0.4
    validation_split: float = 0.2
    patience: int = 8
    seed: int = 42
    optuna_trials: int = 0
    optuna_epochs: int = 6


CONFIG = Config()
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
tf.get_logger().setLevel('ERROR')
tf.keras.utils.set_random_seed(CONFIG.seed)

# Rutas dependientes del material; se fijan en run() vía configure_target().
DATA_ROOT: Path = None
MODEL_PATH: Path = None
HISTORY_PLOT: Path = None
CONFUSION_MATRIX_PLOT: Path = None
OPTUNA_BEST_PARAMS_PATH: Path = None


def configure_target(target: str) -> None:
    global DATA_ROOT, MODEL_PATH, HISTORY_PLOT, CONFUSION_MATRIX_PLOT, OPTUNA_BEST_PARAMS_PATH
    DATA_ROOT = dataset_module.raw_root(target)
    stem = f'ecosort_{target}_tf'
    MODEL_PATH = ARTIFACTS_DIR / f'{stem}.keras'
    HISTORY_PLOT = ARTIFACTS_DIR / f'{stem}_history.png'
    CONFUSION_MATRIX_PLOT = ARTIFACTS_DIR / f'{stem}_confusion_matrix.png'
    OPTUNA_BEST_PARAMS_PATH = ARTIFACTS_DIR / f'{stem}_optuna_best_params.json'


def build_datasets(config: Config) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    common = dict(
        directory=str(DATA_ROOT), labels='inferred', label_mode='binary',
        image_size=config.image_size, batch_size=config.batch_size,
        validation_split=config.validation_split, seed=config.seed,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset='training', shuffle=True, **common)
    valid_ds = tf.keras.utils.image_dataset_from_directory(subset='validation', shuffle=True, **common)
    class_names = train_ds.class_names

    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune).prefetch(autotune)
    valid_ds = valid_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune).prefetch(autotune)
    return train_ds, valid_ds, class_names


def build_cnn(image_size: Tuple[int, int], dropout: float, dense_units: int = 256, augment: bool = True) -> tf.keras.Model:
    """CNN desde cero: 4 bloques conv + GlobalAveragePooling. Sin preentrenar."""
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = inputs
    if augment:
        x = tf.keras.layers.RandomFlip('horizontal')(x)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
        x = tf.keras.layers.RandomContrast(0.1)(x)

    def block(tensor, filters: int, n_convs: int, drop: float):
        for _ in range(n_convs):
            tensor = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(tensor)
            tensor = tf.keras.layers.BatchNormalization()(tensor)
            tensor = tf.keras.layers.ReLU()(tensor)
        tensor = tf.keras.layers.MaxPooling2D()(tensor)
        return tf.keras.layers.Dropout(drop)(tensor)

    x = block(x, 32, 1, 0.10)
    x = block(x, 64, 1, 0.15)
    x = block(x, 128, 2, 0.20)
    x = block(x, 256, 2, 0.30)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='ecosort')


def class_weight_from_dirs(class_names: List[str]) -> Dict[int, float]:
    counts = [len(dataset_module._list_images(DATA_ROOT / name)) for name in class_names]
    total = sum(counts) or 1
    return {i: total / (len(counts) * c) for i, c in enumerate(counts) if c > 0}


def get_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return {
        'adam': tf.keras.optimizers.Adam(learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate, momentum=0.9),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate),
    }[name]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for target, prediction in zip(y_true, y_pred):
        matrix[int(target), int(prediction)] += 1
    return matrix


def train_model(config: Config, train_ds, valid_ds, class_names, hyperparams, epochs, callbacks=None):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    model = build_cnn(config.image_size, float(hyperparams['dropout']),
                      int(hyperparams['dense_units']), bool(hyperparams['augment']))
    model.compile(
        optimizer=get_optimizer(str(hyperparams['optimizer']), float(hyperparams['learning_rate'])),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    if callbacks is None:
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=max(2, config.patience // 2), restore_best_weights=True)]
    history = model.fit(
        train_ds, validation_data=valid_ds, epochs=epochs, callbacks=callbacks,
        class_weight=class_weight_from_dirs(class_names), verbose=0,
    )
    return model, history


def evaluate(model: tf.keras.Model, dataset, class_names: List[str]) -> Dict[str, object]:
    prob_batches, true_batches = [], []
    for features, labels in dataset:
        prob_batches.append(model.predict(features, verbose=0).ravel())
        true_batches.append(labels.numpy().ravel())
    probabilities = np.concatenate(prob_batches)
    true = np.concatenate(true_batches).astype(int)
    predicted = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(true, predicted)
    accuracy = float((predicted == true).mean())
    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(true, probabilities)
    auc = float(auc_metric.result().numpy())
    tp = int(((predicted == 1) & (true == 1)).sum())
    fp = int(((predicted == 1) & (true == 0)).sum())
    fn = int(((predicted == 0) & (true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicho'); ax.set_ylabel('Real'); ax.set_title('Matriz de confusión')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(CONFUSION_MATRIX_PLOT, dpi=120); plt.close(fig)

    print(f'\nClases: {class_names}')
    print('Matriz de confusión (filas=real, columnas=predicho):')
    print(matrix)
    print(f'Accuracy: {accuracy:.4f} | AUC: {auc:.4f} | precision(+): {precision:.4f} | recall(+): {recall:.4f}')
    print('Matriz de confusión guardada en', CONFUSION_MATRIX_PLOT)
    return {'accuracy': accuracy, 'auc': auc, 'precision': precision, 'recall': recall,
            'confusion_matrix': matrix, 'class_names': class_names}


def plot_history(history, save_path: Path) -> None:
    metrics = history.history
    epochs = np.arange(1, len(metrics['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, metrics['loss'], label='train')
    axes[0].plot(epochs, metrics['val_loss'], label='valid')
    axes[0].set_title('loss'); axes[0].set_xlabel('epoch'); axes[0].legend()
    axes[1].plot(epochs, metrics['accuracy'], label='train')
    axes[1].plot(epochs, metrics['val_accuracy'], label='valid')
    axes[1].set_title('accuracy'); axes[1].set_xlabel('epoch'); axes[1].legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
    print('Gráfica guardada en', save_path)


def run_optuna(config: Config, train_ds, valid_ds, class_names):
    if optuna is None:
        raise SystemExit('Optuna no está instalado. Instálalo con: pip install optuna')
    print('Ejecutando Optuna para TensorFlow...')

    def objective(trial):
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.15, 0.5),
            'dense_units': trial.suggest_categorical('dense_units', [64, 128, 256]),
            'augment': trial.suggest_categorical('augment', [True, False]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
        }
        _, history = train_model(config, train_ds, valid_ds, class_names, hyperparams, config.optuna_epochs)
        val_auc = history.history.get('val_auc', [])  # AUC: robusto también para el modelo desequilibrado
        return float(max(val_auc)) if val_auc else 0.0

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='ecosort_tf')
    study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)
    return study


def run(target: str, epochs: int = CONFIG.epochs, optuna_trials: int = CONFIG.optuna_trials,
        optuna_epochs: int = CONFIG.optuna_epochs) -> Dict[str, object]:
    configure_target(target)
    gpus = tf.config.list_physical_devices('GPU')
    print(f'== EcoSort TF | material: {target} | GPU: {"sí" if gpus else "no (CPU)"} ==')

    if not dataset_module.is_ready(target):
        raise SystemExit(f'No hay dataset para {target}. Corre: python dataset.py --target {target}')

    train_ds, valid_ds, class_names = build_datasets(CONFIG)

    if optuna_trials > 0:
        cfg = Config(optuna_trials=optuna_trials, optuna_epochs=optuna_epochs)
        study = run_optuna(cfg, train_ds, valid_ds, class_names)
        hyperparams = study.best_params
        print('\nMejores hiperparámetros (Optuna):')
        for key, value in hyperparams.items():
            print(f'  {key}: {value}')
        OPTUNA_BEST_PARAMS_PATH.write_text(json.dumps(hyperparams, indent=2))
    else:
        hyperparams = {'learning_rate': CONFIG.learning_rate, 'dropout': CONFIG.dropout,
                       'dense_units': 256, 'augment': True, 'optimizer': 'adam'}

    final_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0),
    ]
    model, history = train_model(CONFIG, train_ds, valid_ds, class_names, hyperparams, epochs, callbacks=final_callbacks)
    model.save(MODEL_PATH)   # <-- se guarda de verdad (bug corregido)
    plot_history(history, HISTORY_PLOT)
    results = evaluate(model, valid_ds, class_names)
    print('\nModelo guardado en', MODEL_PATH)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='EcoSort — entrena un detector de material (TF, desde cero).')
    parser.add_argument('--target', choices=dataset_module.ALL_TARGETS, required=True)
    parser.add_argument('--epochs', type=int, default=CONFIG.epochs)
    parser.add_argument('--optuna-trials', type=int, default=CONFIG.optuna_trials)
    parser.add_argument('--optuna-epochs', type=int, default=CONFIG.optuna_epochs)
    args = parser.parse_args()
    run(args.target, epochs=args.epochs, optuna_trials=args.optuna_trials, optuna_epochs=args.optuna_epochs)


if __name__ == '__main__':
    main()

"""Entrena el detector perro vs no-perro con una CNN **desde cero** (sin preentrenar).

Lee las imágenes de `data/dog_vs_notdog/raw/{dog,not_dog}` y entrena una CNN con
data augmentation, `class_weight`, early stopping y LR scheduling. Evalúa sobre la
validación, guarda el modelo y una gráfica de curvas.

Usa GPU automáticamente si TensorFlow la detecta; si no, corre en CPU (más lento).

Uso:
    python train.py                 # entrena 30 epochs
    python train.py --epochs 40
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use('Agg')  # backend sin ventana: guarda la gráfica a archivo
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import dataset as dataset_module


@dataclass(frozen=True)
class Config:
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    dropout: float = 0.4
    validation_split: float = 0.2
    patience: int = 10
    seed: int = 42


CONFIG = Config()
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = dataset_module.RAW_ROOT
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
TENSORBOARD_DIR = ARTIFACTS_DIR / 'tensorboard'
MODEL_PATH = ARTIFACTS_DIR / 'dog_detector_tf.keras'
HISTORY_PLOT = ARTIFACTS_DIR / 'dog_detector_tf_history.png'
for directory in (ARTIFACTS_DIR, TENSORBOARD_DIR):
    directory.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(CONFIG.seed)


def build_datasets(config: Config) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
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
    # Ambas con shuffle=True y el mismo seed: así el split train/val es consistente.
    train_ds = tf.keras.utils.image_dataset_from_directory(subset='training', shuffle=True, **common)
    valid_ds = tf.keras.utils.image_dataset_from_directory(subset='validation', shuffle=True, **common)
    class_names = train_ds.class_names

    # Sin .cache(): cachear miles de imágenes en RAM causa OOM en máquinas modestas.
    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune).prefetch(autotune)
    valid_ds = valid_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune).prefetch(autotune)
    return train_ds, valid_ds, class_names


def build_cnn(image_size: Tuple[int, int], dropout: float, augment: bool = True) -> tf.keras.Model:
    """CNN desde cero: 4 bloques (hasta 256 filtros) + GlobalAveragePooling.

    Con doble conv en los bloques profundos para captar más detalle y dropout
    creciente + data augmentation para combatir el sobreajuste. GlobalAveragePooling
    (en vez de Flatten) mantiene el uso de VRAM bajo. Sin preentrenar.
    """
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = inputs
    if augment:
        x = tf.keras.layers.RandomFlip('horizontal')(x)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)
        x = tf.keras.layers.RandomContrast(0.1)(x)

    def conv_block(tensor, filters: int, n_convs: int, drop: float):
        for _ in range(n_convs):
            tensor = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(tensor)
            tensor = tf.keras.layers.BatchNormalization()(tensor)
            tensor = tf.keras.layers.ReLU()(tensor)
        tensor = tf.keras.layers.MaxPooling2D()(tensor)
        return tf.keras.layers.Dropout(drop)(tensor)

    x = conv_block(x, 32, 1, 0.10)    # 224 -> 112
    x = conv_block(x, 64, 1, 0.15)    # 112 -> 56
    x = conv_block(x, 128, 2, 0.20)   # 56 -> 28
    x = conv_block(x, 256, 2, 0.30)   # 28 -> 14

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='dog_detector')


def class_weight_from_dirs(class_names: List[str]) -> Dict[int, float]:
    counts = [len(dataset_module._list_images(DATA_ROOT / name)) for name in class_names]
    total = sum(counts) or 1
    return {i: total / (len(counts) * c) for i, c in enumerate(counts) if c > 0}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for target, prediction in zip(y_true, y_pred):
        matrix[int(target), int(prediction)] += 1
    return matrix


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset, class_names: List[str]) -> Dict[str, object]:
    prob_batches, true_batches = [], []
    for features, labels in dataset:
        prob_batches.append(model.predict(features, verbose=0).ravel())
        true_batches.append(labels.numpy().ravel())
    probabilities = np.concatenate(prob_batches)
    predicted = (probabilities >= 0.5).astype(int)
    true = np.concatenate(true_batches).astype(int)

    matrix = confusion_matrix(true, predicted)
    accuracy = float((predicted == true).mean())
    print(f'\nClases: {class_names}')
    print('Matriz de confusión (filas=real, columnas=predicho):')
    print(matrix)
    print(f'Accuracy en validación: {accuracy:.4f}')
    return {'accuracy': accuracy, 'confusion_matrix': matrix, 'class_names': class_names}


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


def run(epochs: int = CONFIG.epochs) -> Dict[str, object]:
    """Entrena, evalúa y guarda modelo + gráfica. Devuelve las métricas finales."""
    gpus = tf.config.list_physical_devices('GPU')
    print('TensorFlow:', tf.__version__, '| GPU:', 'sí' if gpus else 'no (CPU)')

    if not dataset_module.is_ready():
        raise SystemExit('No hay dataset. Corre primero: python dataset.py')

    train_ds, valid_ds, class_names = build_datasets(CONFIG)
    class_weight = class_weight_from_dirs(class_names)
    print('Clases:', class_names, '| class_weight:', class_weight)

    model = build_cnn(CONFIG.image_size, CONFIG.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(TENSORBOARD_DIR / 'dog_detector')),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=CONFIG.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(
        train_ds, validation_data=valid_ds, epochs=epochs, callbacks=callbacks, class_weight=class_weight
    )

    plot_history(history, HISTORY_PLOT)
    results = evaluate(model, valid_ds, class_names)
    print('\nModelo guardado en', MODEL_PATH)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Entrena el detector perro vs no-perro (desde cero).')
    parser.add_argument('--epochs', type=int, default=CONFIG.epochs, help='Número de epochs.')
    args = parser.parse_args()
    run(epochs=args.epochs)


if __name__ == '__main__':
    main()

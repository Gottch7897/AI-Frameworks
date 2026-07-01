"""TensorFlow: Wine MLP and FashionMNIST prototype.

This module keeps the notebook logic in script form so the project can evolve
without changing the core data, model, training, or evaluation flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
PROJECT_ROOT = Path('.').resolve()
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts' / 'tensorflow'
CHECKPOINT_DIR = ARTIFACTS_DIR / 'checkpoints'
TENSORBOARD_DIR = ARTIFACTS_DIR / 'tensorboard'
for directory in (ARTIFACTS_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR):
    directory.mkdir(parents=True, exist_ok=True)


tf.keras.utils.set_random_seed(CONFIG.seed)


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


def standardize(train_features: np.ndarray, valid_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-8
    return (train_features - mean) / std, (valid_features - mean) / std


def build_wine_datasets(config: TrainConfig):
    features, labels, metadata = load_wine_arrays()
    train_features, valid_features, train_labels, valid_labels = split_arrays(
        features, labels, config.validation_split, config.seed
    )
    train_features, valid_features = standardize(train_features, valid_features)

    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_features, valid_labels))

    train_ds = train_ds.shuffle(buffer_size=len(train_features), seed=config.seed).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, metadata


def build_wine_mlp(input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='wine_mlp')


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    return model


def plot_history(history: tf.keras.callbacks.History, title: str) -> None:
    metrics = history.history
    epochs = np.arange(1, len(metrics['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, metrics['loss'], label='train')
    axes[0].plot(epochs, metrics['val_loss'], label='valid')
    axes[0].set_title(f'{title} - loss')
    axes[0].set_xlabel('epoch')
    axes[0].legend()

    axes[1].plot(epochs, metrics['accuracy'], label='train')
    axes[1].plot(epochs, metrics['val_accuracy'], label='valid')
    axes[1].set_title(f'{title} - accuracy')
    axes[1].set_xlabel('epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def build_fashion_loaders(batch_size: int = 64):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype('float32') / 255.0)[..., np.newaxis]
    x_test = (x_test.astype('float32') / 255.0)[..., np.newaxis]

    valid_size = int(len(x_train) * 0.1)
    x_valid, y_valid = x_train[:valid_size], y_train[:valid_size]
    x_train, y_train = x_train[valid_size:], y_train[valid_size:]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train), seed=CONFIG.seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds, test_ds


def build_fashion_cnn(num_classes: int = 10) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='fashion_cnn')


def compile_fashion_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for target, prediction in zip(y_true, y_pred):
        matrix[target, prediction] += 1
    return matrix


def evaluate_errors(model: tf.keras.Model, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    predictions = model.predict(dataset, verbose=0)
    predicted_labels = predictions.argmax(axis=1)
    true_labels = np.concatenate([labels for _, labels in dataset], axis=0)
    error_indices = np.where(true_labels != predicted_labels)[0].tolist()
    return true_labels, predicted_labels, error_indices


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
    print('TensorFlow version:', tf.__version__)
    print('Artifacts directory:', ARTIFACTS_DIR)
    print('Suggested modular structure:')
    print(PROJECT_TREE)

    train_ds, valid_ds, wine_metadata = build_wine_datasets(CONFIG)
    wine_model = build_wine_mlp(
        input_dim=wine_metadata['num_features'],
        hidden_dim=CONFIG.hidden_dim,
        num_classes=wine_metadata['num_classes'],
        dropout=CONFIG.dropout,
    )
    compile_model(wine_model, CONFIG.learning_rate)

    wine_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ]

    # history = wine_model.fit(
    #     train_ds,
    #     validation_data=valid_ds,
    #     epochs=CONFIG.epochs,
    #     callbacks=wine_callbacks,
    # )
    # plot_history(history, 'Wine MLP')

    fashion_train_ds, fashion_valid_ds, fashion_test_ds = build_fashion_loaders(CONFIG.batch_size)
    fashion_model = build_fashion_cnn()
    compile_fashion_model(fashion_model, CONFIG.learning_rate)

    fashion_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / 'fashion_cnn.keras'),
            monitor='val_loss',
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(TENSORBOARD_DIR / 'fashion_cnn')),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ]

    # fashion_history = fashion_model.fit(
    #     fashion_train_ds,
    #     validation_data=fashion_valid_ds,
    #     epochs=CONFIG.epochs,
    #     callbacks=fashion_callbacks,
    # )

    # y_true, y_pred, error_indices = evaluate_errors(fashion_model, fashion_test_ds)
    # cm = confusion_matrix_numpy(y_true, y_pred, num_classes=10)


if __name__ == '__main__':
    main()

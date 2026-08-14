"""Entrenamiento TensorFlow: variante small con class_weight."""

from __future__ import annotations

import argparse

from train_tf import CONFIG, run

VARIANT = 'tf_small_class_weight'


def main() -> None:
    parser = argparse.ArgumentParser(description='Entrena la variante TensorFlow small con class_weight.')
    parser.add_argument('--epochs', type=int, default=CONFIG.epochs)
    parser.add_argument('--optuna-trials', type=int, default=CONFIG.optuna_trials)
    parser.add_argument('--optuna-epochs', type=int, default=CONFIG.optuna_epochs)
    args = parser.parse_args()
    run(variant=VARIANT, epochs=args.epochs, optuna_trials=args.optuna_trials, optuna_epochs=args.optuna_epochs)


if __name__ == '__main__':
    main()

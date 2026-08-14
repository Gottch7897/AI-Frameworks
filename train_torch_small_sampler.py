"""Entrenamiento PyTorch: variante small con WeightedRandomSampler."""

from __future__ import annotations

import argparse

from train_torch import CONFIG, run

VARIANT = 'torch_small_sampler'


def main() -> None:
    parser = argparse.ArgumentParser(description='Entrena la variante PyTorch small con WeightedRandomSampler.')
    parser.add_argument('--epochs', type=int, default=CONFIG.epochs)
    parser.add_argument('--optuna-trials', type=int, default=CONFIG.optuna_trials)
    parser.add_argument('--optuna-epochs', type=int, default=CONFIG.optuna_epochs)
    args = parser.parse_args()
    run(variant=VARIANT, epochs=args.epochs, optuna_trials=args.optuna_trials, optuna_epochs=args.optuna_epochs)


if __name__ == '__main__':
    main()

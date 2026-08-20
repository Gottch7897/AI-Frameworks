"""Exporta los modelos EcoSort a ONNX para inferencia portable.

Ejemplos:
    python export_onnx.py --source tf
    python export_onnx.py --source torch
    python export_onnx.py --source all

Los modelos se guardan en artifacts/onnx/{tf,torch}/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_ROOT / 'artifacts'
ONNX_DIR = ARTIFACTS / 'onnx'
MATERIALS = ('plastico', 'vidrio', 'papel')
IMAGE_SIZE = 224


def export_tensorflow() -> None:
    import tensorflow as tf
    import tf2onnx

    output_dir = ONNX_DIR / 'tf'
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = [tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32, name='images')]
    for material in MATERIALS:
        source = ARTIFACTS / f'ecosort_{material}_tf.keras'
        output = output_dir / f'ecosort_{material}.onnx'
        if not source.exists():
            raise SystemExit(f'Falta el modelo {source}')
        model = tf.keras.models.load_model(source)
        tf2onnx.convert.from_keras(model, input_signature=signature, opset=17,
                                    output_path=str(output))
        print(f'Exportado TF: {output}')


def export_pytorch() -> None:
    import torch
    from train_torch import MaterialDetector

    output_dir = ONNX_DIR / 'torch'
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    for material in MATERIALS:
        source = ARTIFACTS / f'ecosort_{material}_torch.pt'
        params_path = ARTIFACTS / f'ecosort_{material}_torch_optuna_best_params.json'
        output = output_dir / f'ecosort_{material}.onnx'
        if not source.exists():
            raise SystemExit(f'Falta el modelo {source}')
        params = json.loads(params_path.read_text()) if params_path.exists() else {}
        model = MaterialDetector(dropout=float(params.get('dropout', 0.4)),
                                 dense_units=int(params.get('dense_units', 256)))
        model.load_state_dict(torch.load(source, map_location='cpu'))
        model.eval()
        torch.onnx.export(model, sample, str(output), input_names=['images'],
                          output_names=['logits'], dynamic_axes={'images': {0: 'batch'},
                          'logits': {0: 'batch'}}, opset_version=18)
        print(f'Exportado Torch: {output}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Exporta modelos EcoSort a ONNX.')
    parser.add_argument('--source', choices=['tf', 'torch', 'all'], default='all')
    args = parser.parse_args()
    if args.source in ('tf', 'all'):
        export_tensorflow()
    if args.source in ('torch', 'all'):
        export_pytorch()


if __name__ == '__main__':
    main()

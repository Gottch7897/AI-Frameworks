"""EcoSort — inferencia compartida: carga los 3 detectores y predice el material.

Cada detector es binario (material vs no-material) y devuelve P(es ese material).
El material predicho es el de mayor probabilidad. Sirve tanto para `predict.py`
(imágenes) como para `camara.py` (cámara con OpenCV).

Funciona con TensorFlow (`framework='tf'`, correr en el venv con tensorflow) o
PyTorch (`framework='torch'`, correr en el venv-torch). La inferencia va en CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_ROOT / 'artifacts'
MATERIALS = ['plastico', 'vidrio', 'papel']
IMG_SIZE = 224


def preprocess(image) -> np.ndarray:
    """A partir de PIL.Image o np.ndarray (RGB) -> (224,224,3) float32 en [0,1]."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(image, dtype=np.float32) / 255.0


def load_models(framework: str = 'tf') -> Dict[str, object]:
    """Carga los 3 modelos de material del framework indicado."""
    models: Dict[str, object] = {}
    if framework == 'tf':
        import tensorflow as tf
        for material in MATERIALS:
            path = ARTIFACTS / f'ecosort_{material}_tf.keras'
            if not path.exists():
                raise SystemExit(f'Falta el modelo {path}. Entrena con: python train_tf.py --target {material}')
            models[material] = tf.keras.models.load_model(path)
    elif framework == 'torch':
        import torch
        from train_torch import MaterialDetector
        for material in MATERIALS:
            path = ARTIFACTS / f'ecosort_{material}_torch.pt'
            if not path.exists():
                raise SystemExit(f'Falta el modelo {path}. Entrena con: python train_torch.py --target {material}')
            net = MaterialDetector()
            net.load_state_dict(torch.load(path, map_location='cpu'))
            net.eval()
            models[material] = net
    else:
        raise ValueError("framework debe ser 'tf' o 'torch'")
    return models


def predict_probs(models: Dict[str, object], framework: str, image) -> Dict[str, float]:
    """Devuelve {material: P(es ese material)} para una imagen."""
    x = preprocess(image)
    probs: Dict[str, float] = {}
    if framework == 'tf':
        batch = x[None, ...]
        for material, model in models.items():
            probs[material] = float(model.predict(batch, verbose=0).ravel()[0])
    else:
        import torch
        tensor = torch.from_numpy(x.transpose(2, 0, 1)[None, ...])
        with torch.no_grad():
            for material, model in models.items():
                probs[material] = float(torch.sigmoid(model(tensor)).item())
    return probs


def classify(probs: Dict[str, float], threshold: float = 0.5) -> Tuple[Optional[str], float]:
    """Material de mayor probabilidad; None si ninguno supera el umbral."""
    best = max(probs, key=probs.get)
    return (best, probs[best]) if probs[best] >= threshold else (None, probs[best])

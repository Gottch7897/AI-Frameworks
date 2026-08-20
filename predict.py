"""EcoSort — consumo del modelo: clasifica el material de una o varias imágenes.

Corre los 3 detectores (plástico / vidrio / papel) sobre cada imagen y reporta el
material más probable, con las 3 probabilidades.

Ejemplos:
    python predict.py foto.jpg
    python predict.py --framework torch foto1.jpg foto2.jpg
    python predict.py --dir alguna_carpeta/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image

import ecosort_infer as infer

VALID_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def gather_paths(images: List[str], directory: str | None) -> List[Path]:
    paths = [Path(p) for p in images]
    if directory:
        paths += [p for p in Path(directory).rglob('*')
                  if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description='EcoSort — clasifica el material de una imagen.')
    parser.add_argument('images', nargs='*', help='Ruta(s) de imagen.')
    parser.add_argument('--framework', choices=['tf', 'torch', 'onnx-tf', 'onnx-torch'], default='tf')
    parser.add_argument('--dir', default=None, help='Clasificar todas las imágenes de una carpeta.')
    args = parser.parse_args()

    paths = gather_paths(args.images, args.dir)
    if not paths:
        raise SystemExit('Pasa una o más rutas de imagen, o --dir CARPETA.')

    models = infer.load_models(args.framework)
    print(f'\nEcoSort ({args.framework}) — detectores: {", ".join(infer.MATERIALS)}\n')
    for path in paths:
        try:
            image = Image.open(path)
        except Exception as error:
            print(f'{path.name}: no se pudo abrir ({error})')
            continue
        probs = infer.predict_probs(models, args.framework, image)
        material, confidence = infer.classify(probs)
        etiqueta = material.upper() if material else 'NINGUNO (no reciclable / desconocido)'
        detalle = '  '.join(f'{m}={probs[m]:.0%}' for m in infer.MATERIALS)
        print(f'{path.name:<28} -> {etiqueta:<10} ({confidence:.0%})   [{detalle}]')


if __name__ == '__main__':
    main()

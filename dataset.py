"""EcoSort — descarga y organiza el dataset de reciclaje (TrashNet).

Construye 3 datasets binarios (uno-vs-resto) para 3 detectores de material:
**plástico**, **vidrio** y **papel**. El de **vidrio** se deja DESEQUILIBRADO a
propósito (1-vs-resto sin balancear) para analizar el efecto del desbalance;
los otros dos quedan balanceados.

Fuente: TrashNet (GitHub, descarga directa **SIN cuenta**).
Diseñado para reproducibilidad: idempotente (no re-descarga) y con `SEED` fijo.

Uso:
    python dataset.py                 # construye los 3
    python dataset.py --target vidrio # construye solo uno
    python dataset.py --force         # reconstruye
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
ECOSORT_ROOT = PROJECT_ROOT / 'data' / 'ecosort'
DOWNLOAD_DIR = PROJECT_ROOT / 'data' / 'downloads'
TRASHNET_URL = 'https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip'
TRASHNET_DIR = DOWNLOAD_DIR / 'dataset-resized'

VALID_SUFFIXES = {'.jpg', '.jpeg', '.png'}
SEED = 42

# material objetivo -> carpeta(s) de TrashNet que cuentan como POSITIVO
MATERIALS: Dict[str, List[str]] = {
    'plastico': ['plastic'],
    'vidrio': ['glass'],
    'papel': ['paper', 'cardboard'],
}
IMBALANCED_TARGET = 'vidrio'   # este se deja SIN balancear (clase minoritaria)
ALL_TARGETS = list(MATERIALS)


def _list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]


def raw_root(target: str) -> Path:
    """Carpeta con las subcarpetas <material>/ y no_<material>/ de ese modelo."""
    return ECOSORT_ROOT / target / 'raw'


def _download_trashnet() -> Path:
    import urllib.request

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DOWNLOAD_DIR / 'trashnet.zip'
    if not (zip_path.exists() and zip_path.stat().st_size > 40_000_000):
        print('[ecosort] descargando TrashNet (~43 MB, una sola vez)...')
        request = urllib.request.Request(TRASHNET_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request, timeout=180) as response, open(zip_path, 'wb') as handle:
            shutil.copyfileobj(response, handle)
    if not TRASHNET_DIR.exists():
        print('[ecosort] extrayendo...')
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(DOWNLOAD_DIR)
    return TRASHNET_DIR


def counts(target: str) -> Dict[str, int]:
    root = raw_root(target)
    return {d.name: len(_list_images(d)) for d in sorted(root.glob('*')) if d.is_dir()}


def is_ready(target: str) -> bool:
    root = raw_root(target)
    pos, neg = root / target, root / f'no_{target}'
    return pos.exists() and neg.exists() and bool(_list_images(pos)) and bool(_list_images(neg))


def build_target(target: str, force: bool = False) -> Dict[str, int]:
    """Construye el dataset binario <material> vs no_<material>."""
    if is_ready(target) and not force:
        print(f'[ecosort:{target}] ya está listo: {counts(target)}')
        return counts(target)

    source = _download_trashnet()
    rng = random.Random(SEED)
    positive_classes = MATERIALS[target]

    positives: List[Path] = []
    negatives: List[Path] = []
    for folder in source.iterdir():
        if not folder.is_dir():
            continue
        images = _list_images(folder)
        (positives if folder.name in positive_classes else negatives).extend(images)

    rng.shuffle(positives)
    rng.shuffle(negatives)

    # Balanceo: todos menos el objetivo desequilibrado igualan el nº de positivos.
    if target != IMBALANCED_TARGET:
        negatives = negatives[:len(positives)]

    pos_dir = raw_root(target) / target
    neg_dir = raw_root(target) / f'no_{target}'
    for directory in (pos_dir, neg_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(positives):
        shutil.copy(path, pos_dir / f'{target}_{i:04d}{path.suffix.lower()}')
    for i, path in enumerate(negatives):
        shutil.copy(path, neg_dir / f'no_{i:04d}{path.suffix.lower()}')

    result = counts(target)
    etiqueta = 'DESEQUILIBRADO' if target == IMBALANCED_TARGET else 'balanceado'
    print(f'[ecosort:{target}] conteos finales ({etiqueta}): {result}')
    return result


def build(target: Optional[str] = None, force: bool = False) -> Dict[str, Dict[str, int]]:
    targets = [target] if target else ALL_TARGETS
    return {t: build_target(t, force) for t in targets}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construye los datasets de EcoSort (reciclaje).')
    parser.add_argument('--target', choices=ALL_TARGETS, default=None, help='Un material; por defecto los 3.')
    parser.add_argument('--force', action='store_true', help='Reconstruir aunque ya exista.')
    args = parser.parse_args()
    print(build(target=args.target, force=args.force))

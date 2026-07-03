"""Descarga y organiza el dataset del detector: perro vs no-perro.

Fuente: **Cats vs Dogs (Microsoft)** — descarga directa, SIN cuenta. La clase
positiva son perros y la negativa son gatos.

Diseñado para reproducibilidad:
- **Idempotente**: si las carpetas ya tienen las imágenes, no vuelve a descargar
  ni a reconstruir (detecta y sigue).
- **Determinista**: el muestreo usa un `SEED` fijo, así siempre se eligen las
  mismas imágenes por clase.

Uso:
    python dataset.py            # construye (o detecta) el dataset
    python dataset.py --force    # fuerza reconstrucción
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / 'data' / 'dog_vs_notdog' / 'raw'
DOWNLOAD_DIR = PROJECT_ROOT / 'data' / 'downloads'

CATS_DOGS_URL = (
    'https://download.microsoft.com/download/3/E/1/'
    '3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
)

VALID_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
N_PER_CLASS = 4000
SEED = 42
CLASSES = ('dog', 'not_dog')


def _list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]


def _download(url: str, dest: Path, min_size: int) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > min_size:
        print('[dataset] descarga ya presente:', dest)
        return dest
    print('[dataset] descargando (~800 MB, una sola vez):', url)
    urllib.request.urlretrieve(url, dest)
    return dest


def _clean_reencode(folder: Path) -> None:
    """Re-codifica a JPEG limpio y descarta lo irrecuperable.

    El set de Microsoft trae archivos que algunos decoders (TensorFlow) rechazan;
    re-guardarlos con PIL produce JPEGs estándar y elimina los corruptos.
    """
    from PIL import Image
    from tqdm.auto import tqdm

    for image_path in tqdm(_list_images(folder), desc=f'validando {folder.name}'):
        try:
            with Image.open(image_path) as image:
                image.convert('RGB').save(image_path, 'JPEG', quality=95)
        except Exception:
            image_path.unlink(missing_ok=True)


def counts() -> Dict[str, int]:
    return {c: len(_list_images(RAW_ROOT / c)) for c in CLASSES}


def is_ready(n_per_class: int = N_PER_CLASS) -> bool:
    """True si cada clase ya tiene al menos el 90% de las imágenes esperadas."""
    threshold = int(n_per_class * 0.9)
    return all((RAW_ROOT / c).exists() and len(_list_images(RAW_ROOT / c)) >= threshold for c in CLASSES)


def build(n_per_class: int = N_PER_CLASS, seed: int = SEED, force: bool = False) -> Dict[str, int]:
    """Construye (o detecta) el dataset perro vs no-perro. Devuelve conteos."""
    if is_ready(n_per_class) and not force:
        print('[dataset] ya está listo, no se re-descarga. Conteos:', counts())
        return counts()

    rng = random.Random(seed)

    zip_path = _download(CATS_DOGS_URL, DOWNLOAD_DIR / 'kagglecatsanddogs.zip', min_size=700_000_000)
    petimages = DOWNLOAD_DIR / 'PetImages'
    if not (petimages / 'Dog').exists():
        print('[dataset] extrayendo...')
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(DOWNLOAD_DIR)

    dogs = _list_images(petimages / 'Dog')
    cats = _list_images(petimages / 'Cat')
    rng.shuffle(dogs)
    rng.shuffle(cats)

    dog_dir = RAW_ROOT / 'dog'
    notdog_dir = RAW_ROOT / 'not_dog'
    for directory in (dog_dir, notdog_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def _copy(sources: List[Path], dest: Path, tag: str) -> None:
        for i, src in enumerate(sources[:n_per_class]):
            shutil.copy(src, dest / f'{tag}_{i:05d}{src.suffix.lower()}')

    print(f'[dataset] copiando {n_per_class} perros y {n_per_class} gatos...')
    _copy(dogs, dog_dir, 'dog')
    _copy(cats, notdog_dir, 'cat')

    _clean_reencode(dog_dir)
    _clean_reencode(notdog_dir)

    result = counts()
    print('[dataset] conteos finales:', result)
    for name, total in result.items():
        if total == 0:
            raise RuntimeError(f'[dataset] la clase {name} quedó vacía.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construye el dataset perro vs no-perro.')
    parser.add_argument('--force', action='store_true', help='Reconstruir aunque ya exista.')
    parser.add_argument('--n-per-class', type=int, default=N_PER_CLASS, help='Imágenes por clase.')
    args = parser.parse_args()
    build(n_per_class=args.n_per_class, force=args.force)

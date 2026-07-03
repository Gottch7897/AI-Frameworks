# Detector perro vs no-perro — TensorFlow y PyTorch (desde cero)

Clasificador binario de imágenes — **¿es un perro o no?** — implementado en **ambos
frameworks** con una CNN **desde cero** (sin modelos preentrenados), sobre el mismo
dataset y la misma arquitectura, para comparar TensorFlow vs PyTorch.

- **Dataset:** Cats vs Dogs (Microsoft), descarga directa **sin cuenta**. Positivo =
  perros, negativo = gatos. ~4000 imágenes por clase.
- **Modelo:** CNN de 4 bloques convolucionales (~1.19M params) con data augmentation,
  early stopping y LR scheduling. Idéntica en TF y PyTorch.

## Resultados

| Métrica | TensorFlow | PyTorch |
|---|---|---|
| **Accuracy (val)** | 86.3% | 88.5% |
| **AUC** | 0.932 | 0.956 |
| Parámetros | 1,194,721 | 1,192,993 |
| Dispositivo | GPU (GTX 1650 Ti) | GPU (GTX 1650 Ti) |

Ambos frameworks alcanzan un detector sólido desde cero (~86–88%). La pequeña
diferencia está dentro de la variación normal (split aleatorio y LR scheduling), no
refleja superioridad de un framework sobre otro.

## Requisitos

- Python 3.11–3.12
- Dependencias de `requirements.txt` (TensorFlow, PyTorch, NumPy, etc.).
- GPU **opcional** — acelera mucho, pero todo corre en CPU igual.

## Instalación

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo correr

### Opción A — Notebooks (recomendada)
- `detector_tf.ipynb` → versión **TensorFlow**.
- `detector_torch.ipynb` → versión **PyTorch**.

Abre el que quieras y ejecuta todas las celdas ("Run All"): verifica dependencias,
construye/detecta el dataset, entrena y muestra la matriz de confusión y las curvas.

### Opción B — Scripts
```bash
python dataset.py               # descarga y ordena el dataset (idempotente)
python train_tf.py --epochs 30      # entrena la versión TensorFlow
python train_torch.py --epochs 30   # entrena la versión PyTorch
```

`python dataset.py` es **idempotente**: si el dataset ya está, no vuelve a descargar.

## Estructura

```
dataset.py              # descarga + organiza el dataset (compartido, idempotente)
train_tf.py             # entrenamiento TensorFlow (CNN desde cero)
train_torch.py          # entrenamiento PyTorch (misma CNN)
detector_tf.ipynb       # notebook orquestador — TensorFlow
detector_torch.ipynb    # notebook orquestador — PyTorch
train_tf_gpu.sh         # lanzador GPU opcional (TensorFlow, WSL2)
train_torch_gpu.sh      # lanzador GPU opcional (PyTorch)
requirements.txt
artifacts/
    dog_detector_tf.keras / dog_detector_tf_history.png
    dog_detector_torch.pt / dog_detector_torch_history.png
data/                   # dataset (generado; no versionado)
```

## Notas sobre entornos (GPU)

En CPU, TensorFlow y PyTorch conviven en un mismo entorno sin problema (basta el
`requirements.txt`). Para **GPU** conviene un venv por framework, porque cada uno trae
sus propias librerías CUDA y pueden chocar:

- `venv-gpu` con `tensorflow[and-cuda]` → usar `./train_tf_gpu.sh`.
- `venv-torch` con `torch` (CUDA) → usar `./train_torch_gpu.sh`.

## Reproducibilidad

- El muestreo del dataset usa un **seed fijo** → siempre las mismas imágenes.
- Las semillas de entrenamiento están fijadas. En GPU hay algo de no-determinismo
  (cuDNN), así que los números salen **muy parecidos, no idénticos** entre corridas
  (normal en deep learning). Rango esperado: ~85–89% de accuracy.

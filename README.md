# Detector perro vs no-perro (TensorFlow, desde cero)

Clasificador binario de imágenes — **¿es un perro o no?** — entrenado con una CNN
**desde cero** (sin modelos preentrenados) en TensorFlow/Keras.

- **Dataset:** Cats vs Dogs (Microsoft), descarga directa **sin cuenta**. Positivo =
  perros, negativo = gatos. ~4000 imágenes por clase.
- **Modelo:** CNN de 4 bloques convolucionales con data augmentation, `class_weight`,
  early stopping y LR scheduling.
- **Resultado:** ~**86% de accuracy** y **AUC ~0.93** en validación.

## Requisitos

- Python 3.11–3.12
- Las dependencias de `requirements.txt` (TensorFlow, NumPy, Matplotlib, Pillow, tqdm).
- GPU **opcional** — acelera mucho, pero todo corre en CPU igual.

## Instalación

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo correr

### Opción A — Notebook (recomendada)
Abre `detector.ipynb` y ejecuta todas las celdas ("Run All"). El notebook:
1. Verifica las dependencias.
2. Construye o detecta el dataset (`dataset.py`).
3. Entrena el modelo (`train.py`).
4. Muestra la matriz de confusión y la gráfica de entrenamiento.

### Opción B — Scripts
```bash
python dataset.py          # descarga y ordena el dataset (idempotente)
python train.py --epochs 30   # entrena, evalúa y guarda el modelo
```

`python dataset.py` es **idempotente**: si el dataset ya está, no vuelve a descargar.

## Estructura

```
dataset.py          # descarga + organiza Cats-vs-Dogs (idempotente, seed fijo)
train.py            # CNN desde cero: entrenamiento + evaluación + gráfica
detector.ipynb      # notebook orquestador (deps → dataset → train → resultados)
train_gpu.sh        # lanzador opcional para GPU en WSL2
requirements.txt
artifacts/
    dog_detector.keras            # modelo entrenado
    dog_detector_history.png      # curvas de loss/accuracy
data/                             # dataset (generado; no versionado)
```

## Reproducibilidad

- El muestreo del dataset usa un **seed fijo** → siempre las mismas imágenes.
- Las semillas de entrenamiento están fijadas. **Nota:** en GPU hay algo de
  no-determinismo (cuDNN), así que los números salen **muy parecidos, no idénticos**
  entre corridas — es normal en deep learning. Rango esperado: ~85–87% de accuracy.

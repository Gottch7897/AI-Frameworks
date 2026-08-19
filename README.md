# EcoSort — Clasificación de residuos reciclables (TensorFlow + PyTorch)

Sistema de visión por computador que **clasifica el material de un residuo** a
partir de una imagen o de la cámara: **plástico**, **vidrio** o **papel/cartón**.
Cada material tiene su propio detector binario (uno-vs-resto), implementado en
**TensorFlow y PyTorch** (misma arquitectura, CNN **desde cero**, sin preentrenar)
para comparar frameworks.

## Uso comercial

Pensado para **plantas de reciclaje y basureros inteligentes**: una cámara sobre
la cinta transportadora identifica el material y acciona la separación automática,
reduciendo el trabajo manual y la contaminación entre materiales. También aplicable
a apps ciudadanas de reciclaje (apunta con el celular y te dice a qué contenedor va).

## Los 3 modelos

| Modelo | Positivo | Negativo | Balance |
|---|---|---|---|
| **plástico** | plastic | resto | balanceado (482/482) |
| **vidrio** | glass | resto | **desequilibrado** (501/2026) |
| **papel** | paper + cardboard | resto | balanceado (997/997) |

El modelo de **vidrio se deja desequilibrado a propósito** para estudiar el efecto
del desbalance de clases (ver `INFORME.md`).

## Requisitos e instalación

- Python 3.11–3.12. GPU **opcional** (todo corre en CPU).
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Cómo correr (scripts)

```bash
# 1) Datos: descarga TrashNet y arma los 3 datasets (idempotente, sin cuenta)
python dataset.py

# 2) Entrenar (un modelo por material y framework)
python train_tf.py    --target plastico --epochs 25
python train_torch.py --target vidrio   --epochs 25

# 2b) Con búsqueda de hiperparámetros (Optuna)
python train_tf.py --target papel --epochs 25 --optuna-trials 8 --optuna-epochs 8

# 3) Consumo: clasificar imágenes
python predict.py foto.jpg
python predict.py --framework torch foto.jpg

# 4) Cámara en vivo (OpenCV)
python camara.py                     # webcam
python camara.py --source foto.jpg --save salida.jpg   # probar sin cámara
```

## Estructura

```
dataset.py          # descarga TrashNet + arma los 3 datasets binarios
train_tf.py         # entrenamiento TensorFlow (--target)  + Optuna
train_torch.py      # entrenamiento PyTorch (--target)     + Optuna
ecosort_infer.py    # lógica de inferencia compartida (carga los 3 modelos)
predict.py          # consumo del modelo por imágenes
camara.py           # clasificación en vivo con OpenCV
requirements.txt
INFORME.md          # metodología, resultados, análisis y plan de acción
artifacts/          # modelos entrenados + gráficas + matrices de confusión
data/               # datasets (generados; no versionados)
```

## Reproducibilidad

- El muestreo del dataset usa **seed fijo** → siempre las mismas imágenes.
- Semillas de entrenamiento fijadas. En GPU hay algo de no-determinismo (cuDNN),
  así que los números salen **muy parecidos, no idénticos** entre corridas.
- Los entornos GPU van separados por framework (`venv-gpu` para TF con
  `tensorflow[and-cuda]`, `venv-torch` para PyTorch) para evitar choques de CUDA.

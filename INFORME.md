# EcoSort — Informe de resultados y análisis

## 1. Problema y uso comercial

**EcoSort** clasifica el material de un residuo (**plástico / vidrio / papel**) a
partir de una imagen. El caso comercial es la **separación automática en plantas de
reciclaje**: una cámara sobre la cinta transportadora identifica el material y
acciona el desvío correspondiente, reduciendo el clasificado manual (caro y lento) y
la **contaminación cruzada** entre materiales, que es la principal causa de que
lotes reciclables terminen en el vertedero.

## 2. Metodología

- **Dataset:** TrashNet (imágenes reales de residuos), descarga directa sin cuenta.
- **3 detectores binarios** (uno-vs-resto), cada uno en **TensorFlow y PyTorch** →
  6 modelos, misma arquitectura para comparar frameworks de forma justa.
- **Modelo:** CNN **desde cero** (sin preentrenar): 4 bloques convolucionales +
  GlobalAveragePooling, ~1.2M parámetros, con data augmentation, early stopping y
  LR scheduling.
- **Desbalance:** el modelo de **vidrio** se deja 1:4 a propósito; se compensa con
  `class_weight` (TF) / `pos_weight` (PyTorch).
- **Optuna:** búsqueda automática de hiperparámetros (learning rate, dropout, tamaño
  de la capa densa, optimizador, augmentation).

## 3. Resultados (validación)

| Modelo | Framework | Accuracy | AUC | Precision(+) | Recall(+) |
|---|---|---|---|---|---|
| Plástico | TF | 75.5% | 0.914 | 0.71 | 0.88 |
| Plástico | PyTorch | 75.0% | 0.856 | 0.84 | 0.56 |
| **Vidrio** ⚖️ | TF | 89.1% | 0.947 | 0.72 | 0.81 |
| **Vidrio** ⚖️ | PyTorch | 83.8% | 0.920 | 0.55 | 0.90 |
| Papel | TF | 89.5% | 0.963 | 0.90 | 0.91 |
| Papel | PyTorch | 82.9% | 0.906 | 0.89 | 0.76 |

## 4. Análisis

### 4.1 Por material
- **Papel es el más fácil** (AUC 0.90–0.96): el cartón/papel tiene textura y color
  muy distintivos.
- **Plástico es el más difícil** (AUC 0.86–0.91): es transparente y visualmente
  variado (botellas, envoltorios, colores), se confunde con vidrio. En las
  probabilidades se ve: una botella de plástico activa algo el detector de vidrio.
- **TensorFlow rindió algo mejor** en promedio (papel y plástico), aunque la
  diferencia está dentro de la variación esperable (split y semillas), no indica
  superioridad real de un framework.

### 4.2 El modelo desequilibrado (vidrio) — lo más interesante
Con clases 501 vs 2026, la **accuracy engaña**: predecir siempre "no-vidrio" ya da
~80%. Por eso miramos **AUC (0.92–0.95, bueno)** y sobre todo **precision/recall**:

- **TF quedó más balanceado** (precision 0.72, recall 0.81): buen compromiso.
- **PyTorch quedó agresivo** (recall 0.90, **precision 0.55**): detecta casi todos
  los vidrios, pero **con muchos falsos positivos** (el `pos_weight` sobre-corrige).

Ambos frameworks compensan el desbalance por mecanismos distintos (`class_weight`
en Keras vs `pos_weight` en `BCEWithLogitsLoss`), y eso desplaza el punto de
operación en el trade-off precision/recall.

**Hallazgo clave (varianza):** al re-entrenar el modelo de vidrio con la MISMA
configuración y semilla, la precision/recall cambió bastante (una corrida dio
recall 0.55, otra 0.81) por la **no-determinancia de la GPU (cuDNN)**. Es decir, el
**modelo desequilibrado es inestable entre corridas** — la clase minoritaria y el
umbral fijo de 0.5 lo hacen sensible. **Conclusión:** para un modelo desbalanceado
no hay que confiar en una sola corrida; conviene **promediar varias semillas** y
**ajustar el umbral** en vez de dejarlo en 0.5.

### 4.3 Optuna (búsqueda de hiperparámetros)
Se aplicó Optuna al modelo de **vidrio-TF** (6 trials, 8 epochs por trial, objetivo
= AUC de validación). Mejores hiperparámetros encontrados:
`learning_rate=0.00056, dropout=0.48, dense_units=64, optimizer=adam, augment=True`.

**Comparación con el default manual:**

| vidrio-TF | Accuracy | AUC | Precision | Recall |
|---|---|---|---|---|
| Default (dense=256) | **89.1%** | **0.947** | 0.72 | 0.81 |
| Optuna (dense=64) | 86.1% | 0.894 | 0.74 | 0.56 |

**Hallazgo (importante y honesto):** Optuna **no mejoró** al default; quedó apenas
por debajo. Eligió una capa densa chica (`dense_units=64`) que rinde menos que los
256 por defecto en el entrenamiento completo. Conclusión: **con un presupuesto de
búsqueda chico (6 trials), Optuna no garantiza superar un buen default**; para que
aporte necesita más trials y/o un espacio de búsqueda mejor acotado.

**Lección adicional:** en una prueba con muy pocas epochs por trial (4), Optuna
elegía configuraciones que **subentrenaban** aún más. Subir a 8 epochs por trial
mejora la confiabilidad de la búsqueda, pero sigue sin superar al default aquí.
Moraleja: Optuna es una herramienta, no magia — hay que darle presupuesto suficiente
y validar contra un baseline.

## 5. Plan de acción

1. **Elegir el punto de operación según el costo del negocio.** En reciclaje,
   **dejar pasar vidrio (bajo recall) suele ser peor** que un falso positivo (un
   vidrio mal enviado puede romper y contaminar un lote). → Preferir el modelo de
   **alto recall (PyTorch)** o **bajar el umbral** de decisión del modelo TF.
2. **Mejorar el modelo de plástico** (el más flojo): más datos de plástico,
   augmentation específica (transparencias, reflejos), o distinguir mejor de vidrio.
3. **Completar Optuna** en los 3 materiales y ambos frameworks para exprimir cada
   modelo (por tiempo se hizo en uno representativo).
4. **Balancear vidrio con más datos reales** en vez de solo pesos de clase, para
   subir precision sin sacrificar recall.
5. **Siguiente etapa:** llevar el mejor modelo (TF, convertible a TensorFlow.js) a
   la app IONIC con cámara (OpenCV.js) para el demo final.

## 6. Reproducibilidad y consumo
- Datos y semillas fijas (ver `README.md`).
- Consumo listo: `predict.py` (imágenes) y `camara.py` (cámara en vivo, OpenCV).

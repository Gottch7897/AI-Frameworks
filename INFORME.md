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

Modelos **finales desplegados** (con Optuna donde mejoró). El detalle de la
comparación **baseline vs Optuna** de los 6 modelos está en `Hiperparametros.md`.

| Modelo | Framework | Accuracy | AUC | Precision(+) | Recall(+) |
|---|---|---|---|---|---|
| Plástico | TF | 84.4% | 0.933 | 0.84 | 0.84 |
| Plástico | PyTorch | 79.2% | 0.887 | 0.82 | 0.69 |
| **Vidrio** ⚖️ | TF | ~85–89% | 0.91–0.95 | 0.63 | 0.81 |
| **Vidrio** ⚖️ | PyTorch | 86.5% | 0.946 | 0.60 | 0.90 |
| Papel | TF | 91.0% | 0.979 | 0.94 | 0.89 |
| Papel | PyTorch | 91.5% | 0.969 | 0.91 | 0.93 |

> Nota sobre vidrio-TF: por ser el modelo desbalanceado tiene **alta varianza** entre
> corridas (medimos 89.1% baseline, 85.4% Optuna y 86.9% al re-evaluar) — ver §4.2.

## 4. Análisis

### 4.1 Por material
- **Papel es el más fácil** (AUC 0.90–0.96): el cartón/papel tiene textura y color
  muy distintivos.
- **Plástico es el más difícil** (AUC 0.86–0.91): es transparente y visualmente
  variado (botellas, envoltorios, colores), se confunde con vidrio. En las
  probabilidades se ve: una botella de plástico activa algo el detector de vidrio.
- **TF vs PyTorch quedó parejo:** TF gana en plástico, PyTorch iguala o supera en
  papel y vidrio. Las diferencias están dentro de la variación esperable (split,
  semillas, no-determinancia de GPU) y **no indican superioridad real** de un
  framework — que es justo lo que se espera al usar la misma arquitectura.

### 4.2 El modelo desequilibrado (vidrio) — lo más interesante
Con clases 501 vs 2026, la **accuracy engaña**: predecir siempre "no-vidrio" ya da
~80%. Por eso miramos **AUC (0.92–0.95, bueno)** y sobre todo **precision/recall**:

- Ambos modelos alcanzan **recall alto (~0.81–0.90)**: detectan la mayoría de los
  vidrios, pero con **precision baja (0.60–0.66)** → varios falsos positivos.
- Es el efecto de compensar el desbalance (`class_weight` en Keras / `pos_weight`
  en `BCEWithLogitsLoss`): se sube el recall a costa de la precision.

**Hallazgo clave (varianza):** el modelo de vidrio dio **números distintos en cada
medición** — 89.1% (baseline), 85.4% (Optuna) y 86.9% (al re-evaluar el guardado) —
por la **no-determinancia de la GPU (cuDNN)** y su sensibilidad como clase
minoritaria. **Conclusión:** en un modelo desbalanceado no hay que confiar en una
sola corrida; conviene **promediar varias semillas** y **ajustar el umbral** en vez
de dejarlo en 0.5. (Por eso Optuna no lo mejoró de forma confiable — ver §4.3.)

### 4.3 Optuna (búsqueda de hiperparámetros)
Se aplicó Optuna a **los 6 modelos** (objetivo = AUC de validación). La comparación
completa **baseline vs Optuna** y los mejores hiperparámetros de cada uno están en
**`Hiperparametros.md`**. Resumen:

- **Optuna ayudó claramente** en plástico (TF **+8.9 pp** de accuracy) y en papel,
  y mejoró los 3 modelos de PyTorch.
- **No ayudó en vidrio-TF** (el desbalanceado): eligió una capa densa chica
  (`dense_units=64`) y quedó igual o por debajo del baseline. Por eso, para vidrio-TF
  se recomienda conservar la configuración baseline.

**Moraleja (para la defensa):** Optuna es una herramienta, no magia. Ayuda cuando hay
margen y presupuesto de búsqueda suficiente, pero en un modelo **desbalanceado e
inestable** (vidrio) no garantiza mejora — hay que validarlo contra un baseline y con
varias semillas, no confiar en una sola corrida.

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

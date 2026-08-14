# Comparativa tecnica de modelos PyTorch (dog vs not_dog)

## Fuentes analizadas

- artifacts/dog_detector_torch_torch_deep_focal_metrics.json
- artifacts/dog_detector_torch_torch_deep_focal_confusion_matrix.png
- artifacts/dog_detector_torch_torch_deep_focal_history.png
- artifacts/dog_detector_torch_torch_medium_pos_weight_metrics.json
- artifacts/dog_detector_torch_torch_medium_pos_weight_confusion_matrix.png
- artifacts/dog_detector_torch_torch_medium_pos_weight_history.png
- artifacts/dog_detector_torch_torch_small_sampler_metrics.json
- artifacts/dog_detector_torch_torch_small_sampler_confusion_matrix.png
- artifacts/dog_detector_torch_torch_small_sampler_history.png
- Implementacion: train_torch.py

## Convencion de etiquetas usada en esta corrida

Corrida actualizada con clase positiva = dog.

- clase 0 = not_dog
- clase 1 = dog
- precision, recall y f1 se reportan para dog

## Resumen cuantitativo principal

| Variante | Accuracy | Precision (dog) | Recall (dog) | F1 (dog) | AUC |
|---|---:|---:|---:|---:|---:|
| torch_medium_pos_weight | **0.8956** | 0.9270 | **0.8617** | **0.8932** | **0.9602** |
| torch_deep_focal | 0.8386 | **0.9600** | 0.7111 | 0.8170 | 0.9537 |
| torch_small_sampler | 0.6529 | 0.7179 | 0.5185 | 0.6022 | 0.7205 |

## Analisis por matriz de confusion

### 1) torch_medium_pos_weight (mejor global)
Matriz:

- [[734, 55],
-  [112, 698]]

Lectura tecnica:

- Menor error total: 167/1599.
- Alta cobertura de perros reales: recall dog = 698/(698+112)=0.8617.
- Buena precision en prediccion de dog: 698/(698+55)=0.9270.
- Buen equilibrio entre falsos positivos y falsos negativos.

### 2) torch_deep_focal
Matriz:

- [[765, 24],
-  [234, 576]]

Lectura tecnica:

- Error total: 258/1599.
- Precision de dog sobresaliente (0.9600), casi no etiqueta no perros como perros.
- Debilidad principal: recall de dog moderado (0.7111), deja escapar 234 perros.
- Perfil conservador para clase dog.

### 3) torch_small_sampler
Matriz:

- [[624, 165],
-  [390, 420]]

Lectura tecnica:

- Peor error total: 555/1599.
- Recall de dog bajo (0.5185), pierde demasiados perros reales.
- Tambien incrementa falsos positivos de dog (165).
- Rendimiento claramente inferior en esta corrida.

## Analisis del history (dinamica de entrenamiento)

### torch_medium_pos_weight
- Entrenamiento de 30 epochs.
- Curva train_loss decreciente y valid_loss ruidosa pero con tendencia descendente al final.
- Valid accuracy en rango alto (cerca de 0.85-0.89 en tramo final).

Interpretacion: buena convergencia y mejor generalizacion final en esta ejecucion.

### torch_deep_focal
- Entrenamiento de 30 epochs.
- Loss de train y valid con descenso progresivo, con oscilaciones de validacion.
- Valid accuracy crece de forma sostenida pero con caida marcada en el ultimo punto.

Interpretacion: modelo robusto y preciso para dog, pero menos sensible que pos_weight para capturar todos los perros.

### torch_small_sampler
- Early stopping temprano (7 epochs).
- Divergencia train/valid: train mejora, valid empeora y oscila.
- Valid accuracy termina baja respecto al resto.

Interpretacion: inestabilidad de generalizacion y ajuste insuficiente para este dataset en esta configuracion.

## Marco teorico de cada estrategia de desbalance

### Pos-weight BCE
En BCEWithLogitsLoss(pos_weight=w), el termino de la clase positiva se multiplica por w.

- Incrementa el costo de fallar dog.
- Tiende a mejorar recall de dog cuando dog es clase positiva.
- Puede introducir mas falsos positivos si w es excesivo.

### WeightedRandomSampler
Rebalancea la frecuencia de muestras por clase en el DataLoader.

- Aumenta exposicion a la clase minoritaria.
- Puede elevar varianza entre minibatches y volver inestable validacion.
- Su efectividad depende mucho del tamano y ruido del dataset.

### Focal Loss
Formula binaria:

- FL(pt) = -alpha * (1-pt)^gamma * log(pt)

Efecto:

- Reduce peso de ejemplos faciles.
- Concentra gradiente en ejemplos dificiles.
- Suele subir precision en fronteras ambiguas, a costa de recall si queda muy conservador.

## Implementacion: diferencias entre variantes en train_torch.py

- torch_small_sampler
  - Arquitectura pequena: filtros (32, 64, 128).
  - Desbalance en muestreo: WeightedRandomSampler.
  - Loss: BCEWithLogitsLoss estandar.

- torch_medium_pos_weight
  - Arquitectura intermedia: filtros (32, 64, 128, 192).
  - Loss: BCEWithLogitsLoss con pos_weight = neg/pos para dog.

- torch_deep_focal
  - Arquitectura mas profunda: filtros (32, 64, 128, 256).
  - Loss personalizada: BinaryFocalLoss(gamma=2.0, alpha=0.25).
  - Dropout final mayor (0.45).

## Conclusion actualizada: mejor modelo Torch

Mejor modelo global en esta corrida: torch_medium_pos_weight.

Justificacion tecnica:

- Mejor accuracy (0.8956), recall dog (0.8617), f1 dog (0.8932) y auc (0.9602).
- Menor cantidad total de errores en matriz de confusion.
- Mejor compromiso precision-recall para objetivo detector de perros.

## Lectura por objetivo de negocio

- Si priorizas no perder perros (recall alto de dog): torch_medium_pos_weight.
- Si priorizas minimizar falsas alarmas al marcar dog (precision muy alta): torch_deep_focal.

## Recomendacion operativa

- Usar torch_medium_pos_weight como baseline productivo actual.
- Mantener torch_deep_focal como alternativa conservadora para escenarios sensibles a falsos positivos.
- Como siguiente iteracion, calibrar umbral de decision para optimizar segun costo real (FN vs FP) y no depender solo de threshold 0.5.

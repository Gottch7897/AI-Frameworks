# Instrucciones de ejecución

## Preparación

1. Crea y activa el entorno virtual:

```bash
python -m venv venv
source venv/bin/activate
```

2. Instala dependencias:

```bash
pip install -r requirements.txt
```

3. Verifica que PyTorch vea la GPU:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

4. Construye el dataset si hace falta:

```bash
python dataset.py
```

## Cómo correr cada modelo

### TensorFlow

```bash
python train_tf_small_class_weight.py
python train_tf_medium_focal.py
python train_tf_deep_oversample.py
```

### PyTorch

```bash
python train_torch_small_sampler.py
python train_torch_medium_pos_weight.py
python train_torch_deep_focal.py
```

## Qué hace cada script

| Archivo | Framework | Técnica de desbalance | Salida principal |
|---|---|---|---|
| `train_tf_small_class_weight.py` | TensorFlow | `class_weight` | `.keras`, curvas, matriz de confusión, métricas JSON |
| `train_tf_medium_focal.py` | TensorFlow | Focal Loss | `.keras`, curvas, matriz de confusión, métricas JSON |
| `train_tf_deep_oversample.py` | TensorFlow | Oversampling con `tf.data` | `.keras`, curvas, matriz de confusión, métricas JSON |
| `train_torch_small_sampler.py` | PyTorch | `WeightedRandomSampler` | `.pt`, curvas, matriz de confusión, métricas JSON |
| `train_torch_medium_pos_weight.py` | PyTorch | `pos_weight` | `.pt`, curvas, matriz de confusión, métricas JSON |
| `train_torch_deep_focal.py` | PyTorch | Focal Loss | `.pt`, curvas, matriz de confusión, métricas JSON |

## Progreso en terminal

Los scripts de PyTorch imprimen al inicio si usan GPU o CPU y muestran el avance por epoch en la terminal. Si ves líneas como `Epoch 01/30`, el entrenamiento está activo.

## Resultados

Cada corrida guarda archivos en `artifacts/` con el nombre de la variante. Revisa:

- `*_metrics.json` para accuracy, precision, recall, F1 y AUC.
- `*_confusion_matrix.png` para la matriz de confusión.
- `*_history.png` para las curvas de entrenamiento.

## Nota

Si quieres ejecutar todas las variantes de un framework, usa los wrappers uno por uno. Así los artefactos quedan diferenciados y más fáciles de comparar.

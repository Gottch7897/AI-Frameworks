## Torch
Mejores hiperparámetros (Optuna, Torch Papel):
  learning_rate: 0.0003646439558980723
  dropout: 0.33994362910538695
  dense_units: 128
  optimizer: adam
  Accuracy: 0.9146 | AUC: 0.9687 | precision(+): 0.9078 | recall(+): 0.9257

Mejores hiperparámetros (Optuna, Torch Vidrio):
  learning_rate: 0.00017541893487450815
  dropout: 0.3233119185389446
  dense_units: 128
  optimizer: adam
  Accuracy: 0.8653 | AUC: 0.9458 | precision(+): 0.6027 | recall(+): 0.8980

Mejores hiperparámetros (Optuna, Torch Plastico):
  learning_rate: 0.00017541893487450815
  dropout: 0.3233119185389446
  dense_units: 128
  optimizer: adam
  Accuracy: 0.7917 | AUC: 0.8867 | precision(+): 0.8243 | recall(+): 0.6932

## Tensor

Mejores hiperparámetros (Optuna, TensorFlow Papel):
  learning_rate: 0.00017345566642360953
  dropout: 0.39963567552804824
  dense_units: 256
  augment: False
  optimizer: adam
  Accuracy: 0.9095 | AUC: 0.9788 | precision(+): 0.9375 | recall(+): 0.8945

Mejores hiperparámetros (Optuna, TensorFlow Vidrio):
  learning_rate: 0.00010549579566712062
  dropout: 0.2827610125714093
  dense_units: 64
  augment: False
  optimizer: adam
  Accuracy: 0.8535 | AUC: 0.9074 | precision(+): 0.6268 | recall(+): 0.8091

Mejores hiperparámetros (Optuna, Tensorflow Plastico):
  learning_rate: 0.0003076889128625202
  dropout: 0.297962963756816
  dense_units: 256
  augment: False
  optimizer: adam
  Accuracy: 0.8438 | AUC: 0.9325 | precision(+): 0.8438 | recall(+): 0.8438

## Comparacion con la baseline

### TensorFlow

| Material | Metrica | Baseline | Optuna | Cambio |
|---|---:|---:|---:|---:|
| Plastico | Accuracy | 75.5% | **84.38%** | **+8.88 pp** |
| Plastico | AUC | 0.914 | **0.9325** | **+0.0185** |
| Vidrio | Accuracy | **89.1%** | 85.35% | -3.75 pp |
| Vidrio | AUC | **0.947** | 0.9074 | -0.0396 |
| Papel | Accuracy | 89.5% | **90.95%** | **+1.45 pp** |
| Papel | AUC | 0.963 | **0.9788** | **+0.0158** |

Optuna mejora claramente plastico y ligeramente papel, pero empeora vidrio.
Para vidrio se conserva la configuracion baseline.

### PyTorch

| Material | Metrica | Baseline | Optuna | Cambio |
|---|---:|---:|---:|---:|
| Plastico | Accuracy | 75.0% | **79.17%** | **+4.17 pp** |
| Plastico | AUC | 0.856 | **0.8867** | **+0.0307** |
| Vidrio | Accuracy | 83.8% | **86.53%** | **+2.73 pp** |
| Vidrio | AUC | 0.920 | **0.9458** | **+0.0258** |
| Papel | Accuracy | 82.9% | **91.46%** | **+8.56 pp** |
| Papel | AUC | 0.906 | **0.9687** | **+0.0627** |

Optuna mejora los tres modelos PyTorch, especialmente papel.

### Configuraciones recomendadas

- TensorFlow plastico: configuracion de Optuna.
- TensorFlow papel: configuracion de Optuna.
- TensorFlow vidrio: configuracion baseline.
- PyTorch plastico, vidrio y papel: configuracion de Optuna.

La comparacion debe confirmarse con varias semillas, especialmente en vidrio,
porque su dataset esta desbalanceado y las corridas en GPU pueden variar.

"""
En lugar de modificar el dataset, cambia la PROBABILIDAD
de seleccionar cada muestra en cada batch.

La clase minoritaria aparece más seguido en los batches
aunque el dataset sigue igual en disco.

Diferencia con SMOTE y Undersampling: 
  SMOTE => agrega datos nuevos (dataset crece)
  Undersampling => elimina datos (dataset se reduce)
  Ponderada    => mismos datos, distinta frecuencia de muestreo

pip install torch tensorflow scikit-learn
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#  DATASET DESEQUILIBRADO 
X, y = make_classification(
    n_samples    = 1000,
    n_features   = 10,
    weights      = [0.95, 0.05],
    random_state = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset (sin modificar — igual en ambas técnicas):")
print(f"  Clase 0: {(y_train==0).sum()}  |  Clase 1: {(y_train==1).sum()}")


# PYTORCH — WeightedRandomSampler
print("\n" + "─"*48)
print("PYTORCH — WeightedRandomSampler")
print("─"*48)

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, TensorDataset, WeightedRandomSampler
)

X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_te = torch.tensor(X_test,  dtype=torch.float32)
y_te = torch.tensor(y_test,  dtype=torch.long)

#  Calcular peso de cada muestra ─
# Peso de cada CLASE = inverso de su frecuencia
cuenta        = torch.bincount(y_tr)         # [759, 41]
peso_por_clase = 1.0 / cuenta.float()        # [1/759, 1/41]

print(f"\nPeso por clase:")
print(f"  Clase 0: {peso_por_clase[0]:.6f}  (seleccionada con baja probabilidad)")
print(f"  Clase 1: {peso_por_clase[1]:.6f}  (seleccionada con alta probabilidad)")
print(f"  Ratio  : {peso_por_clase[1]/peso_por_clase[0]:.1f}× más probable clase 1")

# Peso de cada MUESTRA individual (según su clase)
peso_por_muestra = peso_por_clase[y_tr]      # (800,) — peso de cada ejemplo

#  Crear el sampler 
sampler = WeightedRandomSampler(
    weights     = peso_por_muestra,
    num_samples = len(peso_por_muestra),  # cuántas muestras por época
    replacement = True                    # permite repetir muestras
)

#  DataLoader con sampler (NO usar shuffle=True con sampler) ─
loader = DataLoader(
    TensorDataset(X_tr, y_tr),
    batch_size = 32,
    sampler    = sampler              # ← reemplaza shuffle=True
)

# Verificar distribución en un batch de ejemplo
X_batch, y_batch = next(iter(loader))
print(f"\nDistribución en un batch de ejemplo (32 muestras):")
print(f"  Clase 0: {(y_batch==0).sum().item():2d}  |  Clase 1: {(y_batch==1).sum().item():2d}")
print(f"  => aproximadamente 50/50 aunque el dataset tiene 95/5")

#  Modelo y entrenamiento 
model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()   # sin class_weight — el sampler ya balancea

for epoch in range(30):
    model.train()
    for X_b, y_b in loader:
        optimizer.zero_grad()
        criterion(model(X_b), y_b).backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(X_te).argmax(dim=1)
    acc   = (preds == y_te).float().mean().item()
    tp    = ((preds == 1) & (y_te == 1)).sum().item()
    fp    = ((preds == 1) & (y_te == 0)).sum().item()
    fn    = ((preds == 0) & (y_te == 1)).sum().item()
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0

print(f"\nResultados:")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Recall    : {rec:.4f}  ← detectó {rec:.0%} de clase 1")
print(f"  Precision : {prec:.4f}")
print(f"  F1-Score  : {f1:.4f}")


# TENSORFLOW — tf.data.Dataset con resample

print("\n" + "─"*48)
print("TENSORFLOW — tf.data con muestreo ponderado")
print("─"*48)

import tensorflow as tf

#  Separar en un dataset por clase ─
idx_clase0 = np.where(y_train == 0)[0]
idx_clase1 = np.where(y_train == 1)[0]

ds_clase0 = tf.data.Dataset.from_tensor_slices(
    (X_train[idx_clase0].astype(np.float32),
     y_train[idx_clase0])
).shuffle(1000).repeat()

ds_clase1 = tf.data.Dataset.from_tensor_slices(
    (X_train[idx_clase1].astype(np.float32),
     y_train[idx_clase1])
).shuffle(100).repeat()

#  Combinar datasets con pesos de muestreo ─
# [0.5, 0.5] => cada batch tiene ~50% de cada clase
dataset_balanceado = tf.data.Dataset.sample_from_datasets(
    datasets = [ds_clase0, ds_clase1],
    weights  = [0.5, 0.5]              # probabilidad de elegir de cada dataset
).batch(32)

# Verificar distribución en un batch
for X_b, y_b in dataset_balanceado.take(1):
    c0 = tf.reduce_sum(tf.cast(y_b == 0, tf.int32)).numpy()
    c1 = tf.reduce_sum(tf.cast(y_b == 1, tf.int32)).numpy()
    print(f"\nDistribución en un batch de ejemplo (32 muestras):")
    print(f"  Clase 0: {c0:2d}  |  Clase 1: {c1:2d}")
    print(f"  => aproximadamente 50/50 aunque el dataset tiene 95/5")

#  Modelo 
model_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2,  activation="softmax"),
])

model_tf.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss      = "sparse_categorical_crossentropy",
    metrics   = [
        "accuracy",
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
    ]
)

# steps_per_epoch: cuántos batches por época
steps_per_epoch = len(y_train) // 32

model_tf.fit(
    dataset_balanceado,
    epochs           = 30,
    steps_per_epoch  = steps_per_epoch,
    verbose          = 0
)

results = model_tf.evaluate(X_test.astype(np.float32), y_test, verbose=0)
_, acc_tf, rec_tf, prec_tf = results
f1_tf = 2*prec_tf*rec_tf / (prec_tf+rec_tf) if (prec_tf+rec_tf) > 0 else 0

print(f"\nResultados:")
print(f"  Accuracy  : {acc_tf:.4f}")
print(f"  Recall    : {rec_tf:.4f}  ← detectó {rec_tf:.0%} de clase 1")
print(f"  Precision : {prec_tf:.4f}")
print(f"  F1-Score  : {f1_tf:.4f}")


# COMPARATIVA DE LOS 4 ENFOQUES

print("\n" + "─"*48)
print("Comparativa de los 4 enfoques")
print("─"*48)
print(f"\n{'Enfoque':<22} {'Datos cambian':<16} {'Cómo funciona'}")
print("─"*70)
filas = [
    ("SMOTE",            "Sí — crece",     "Genera muestras sintéticas interpolando vecinos"),
    ("Undersampling",    "Sí — reduce",    "Elimina muestras de la clase mayoritaria"),
    ("Class Weight",     "No",             "Multiplica el loss por el peso de la clase"),
    ("Selec. ponderada", "No",             "Aumenta la probabilidad de ver clase minoritaria"),
]
for nombre, cambia, como in filas:
    print(f"  {nombre:<20} {cambia:<16} {como}")

print("""
Cuándo elegir selección ponderada:
  - Quieres mantener todos los datos originales intactos
  - El dataset cabe en memoria (los datos no se copian)
  - Quieres controlar exactamente qué proporción de clases
    ve el modelo en cada batch (ej: 70/30 en vez de 50/50)
  - Usas PyTorch — WeightedRandomSampler es muy eficiente
""")

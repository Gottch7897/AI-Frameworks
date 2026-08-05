"""
Reduce la clase MAYORITARIA hasta igualar la minoritaria.
Opuesto a SMOTE: en lugar de generar datos nuevos, elimina.

Técnicas incluidas:
  1. Random Undersampling  — elimina muestras al azar
  2. Tomek Links           — elimina las más ambiguas de la frontera
  3. NearMiss              — elimina las más lejanas de la frontera

Requiere: imbalanced-learn torch tensorflow scikit-learn
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

print("Dataset original (SIN modificar):")
print(f"  Clase 0: {(y_train==0).sum()}  |  Clase 1: {(y_train==1).sum()}")


# TÉCNICAS DE UNDERSAMPLING
from imblearn.under_sampling import (
    RandomUnderSampler,
    TomekLinks,
    NearMiss,
)

print("\n" + "─"*45)
print("TÉCNICAS DE UNDERSAMPLING")
print("─"*45)

# 1. Random Undersampling — el más simple
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)
print(f"\n1. RandomUnderSampler:")
print(f"   Clase 0: {(y_rus==0).sum()}  |  Clase 1: {(y_rus==1).sum()}")
print(f"   Total muestras: {len(y_rus)}  (antes: {len(y_train)})")

# 2. Tomek Links — elimina solo los pares ambiguos en la frontera
#    Más conservador: no fuerza balance total
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X_train, y_train)
print(f"\n2. TomekLinks (limpieza de frontera):")
print(f"   Clase 0: {(y_tl==0).sum()}  |  Clase 1: {(y_tl==1).sum()}")
print(f"   Total muestras: {len(y_tl)}  (eliminó los más ambiguos)")

# 3. NearMiss — elimina los más lejanos a la clase minoritaria
nm = NearMiss(version=1)
X_nm, y_nm = nm.fit_resample(X_train, y_train)
print(f"\n3. NearMiss (elimina más lejanos):")
print(f"   Clase 0: {(y_nm==0).sum()}  |  Clase 1: {(y_nm==1).sum()}")
print(f"   Total muestras: {len(y_nm)}")

# Usamos RandomUnderSampler para los ejemplos de entrenamiento
X_bal, y_bal = X_rus, y_rus


# PYTORCH
print("\n" + "─"*45)
print("PYTORCH — entrenando con datos undersampled")
print("─"*45)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X_tr = torch.tensor(X_bal,  dtype=torch.float32)
y_tr = torch.tensor(y_bal,  dtype=torch.long)
X_te = torch.tensor(X_test, dtype=torch.float32)
y_te = torch.tensor(y_test, dtype=torch.long)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)

# Batch size pequeño porque el dataset es pequeño ahora
model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()   # sin class_weight — ya está balanceado

for epoch in range(50):             # más épocas porque hay menos datos
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
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

print(f"Train samples : {len(y_bal)}  (reducido desde {len(y_train)})")
print(f"Accuracy      : {acc:.4f}")
print(f"Recall        : {rec:.4f}  ← detectó {rec:.0%} de clase 1")
print(f"Precision     : {prec:.4f}")
print(f"F1-Score      : {f1:.4f}")


# TENSORFLOW
print("\n" + "─"*45)
print("TENSORFLOW — entrenando con datos undersampled")
print("─"*45)

import tensorflow as tf

model_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation="softmax"),
])

model_tf.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss      = "sparse_categorical_crossentropy",  # sin class_weight
    metrics   = [
        "accuracy",
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
    ]
)

model_tf.fit(
    X_bal, y_bal,        # datos ya balanceados por undersampling
    epochs     = 50,
    batch_size = 16,
    verbose    = 0
)

results       = model_tf.evaluate(X_test, y_test, verbose=0)
_, acc_tf, rec_tf, prec_tf = results
f1_tf = 2*prec_tf*rec_tf / (prec_tf+rec_tf) if (prec_tf+rec_tf) > 0 else 0

print(f"Train samples : {len(y_bal)}  (reducido desde {len(y_train)})")
print(f"Accuracy      : {acc_tf:.4f}")
print(f"Recall        : {rec_tf:.4f}  ← detectó {rec_tf:.0%} de clase 1")
print(f"Precision     : {prec_tf:.4f}")
print(f"F1-Score      : {f1_tf:.4f}")


# CUÁNDO USAR CADA TÉCNICA DE UNDERSAMPLING

print("\n" + "─"*45)
print("¿Cuándo usar cada técnica?")
print("─"*45)
tecnicas = [
    ("RandomUnderSampler", "Elimina al azar",
     "Rápido. Puede perder información importante.",
     "> 50:1 y dataset muy grande"),
    ("TomekLinks",         "Elimina pares ambiguos en frontera",
     "Conservador — no fuerza balance total. Limpia el límite de decisión.",
     "Complemento de SMOTE (SMOTETomek)"),
    ("NearMiss",           "Elimina los más lejanos a clase minoritaria",
     "Mantiene los más cercanos a la frontera.",
     "Cuando la frontera de decisión es lo más importante"),
    ("ClusterCentroids",   "Reemplaza clusters por su centroide",
     "Más inteligente que random — preserva estructura.",
     "Dataset grande con clusters bien definidos"),
]
for nombre, metodo, ventaja, cuando in tecnicas:
    print(f"\n  {nombre}")
    print(f"    Método  : {metodo}")
    print(f"    Ventaja : {ventaja}")
    print(f"    Usar si : {cuando}")

print("""
Advertencia crítica:
  Undersampling siempre pierde datos. Si la clase mayoritaria
  tiene 10.000 ejemplos y la minoritaria 100, quedas con solo
  200 muestras en total. Para datasets pequeños esto suele
  ser peor que class_weight o focal loss.
""")

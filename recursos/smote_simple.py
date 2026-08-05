"""
Requisitos: imbalanced-learn torch tensorflow scikit-learn
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

#  DATASET DESEQUILIBRADO 
X, y = make_classification(
    n_samples    = 1000,
    n_features   = 10,
    weights      = [0.95, 0.05],   # 950 clase 0, 50 clase 1
    random_state = 42
)

print("Antes de SMOTE:")
print(f"  Clase 0: {(y==0).sum()}  |  Clase 1: {(y==1).sum()}")

#  APLICAR SMOTE 
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

print("\nDespués de SMOTE:")
print(f"  Clase 0: {(y_bal==0).sum()}  |  Clase 1: {(y_bal==1).sum()}")


# PYTORCH
print("\n" + "─"*40)
print("PYTORCH")
print("─"*40)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42
)

# Tensores
X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_te = torch.tensor(X_test,  dtype=torch.float32)
y_te = torch.tensor(y_test,  dtype=torch.long)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

# Modelo
model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Entrenar
for epoch in range(20):
    model.train()
    for X_b, y_b in loader:
        optimizer.zero_grad()
        criterion(model(X_b), y_b).backward()
        optimizer.step()

# Evaluar
model.eval()
with torch.no_grad():
    preds = model(X_te).argmax(dim=1)
    acc   = (preds == y_te).float().mean().item()
    rec   = ((preds == 1) & (y_te == 1)).sum().item() / (y_te == 1).sum().item()

print(f"Accuracy : {acc:.4f}")
print(f"Recall   : {rec:.4f}  ← detectó {rec:.0%} de la clase minoritaria")

# 
# TENSORFLOW
# 
print("\n" + "─"*40)
print("TENSORFLOW")
print("─"*40)

import tensorflow as tf

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42
)

model_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(2,  activation="softmax"),
])
model_tf.compile(
    optimizer = "adam",
    loss      = "sparse_categorical_crossentropy",
    metrics   = ["accuracy", tf.keras.metrics.Recall(name="recall")]
)

model_tf.fit(X_train2, y_train2, epochs=20, batch_size=32, verbose=0)

_, acc_tf, rec_tf = model_tf.evaluate(X_test2, y_test2, verbose=0)
print(f"Accuracy : {acc_tf:.4f}")
print(f"Recall   : {rec_tf:.4f}  ← detectó {rec_tf:.0%} de la clase minoritaria")

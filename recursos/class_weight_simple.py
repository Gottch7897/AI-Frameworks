"""
No modifica los datos — penaliza más los errores
en la clase minoritaria durante el entrenamiento.
requiere: torch tensorflow scikit-learn
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#  DATASET DESEQUILIBRADO 
X, y = make_classification(
    n_samples    = 1000,
    n_features   = 10,
    weights      = [0.95, 0.05],   # 950 clase 0, 50 clase 1
    random_state = 42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset SIN balancear:")
print(f"  Train → Clase 0: {(y_train==0).sum()}  |  Clase 1: {(y_train==1).sum()}")

#  CALCULAR PESOS 
# Fórmula: w_i = N_total / (N_clases × N_clase_i)
n_total  = len(y_train)
n_clase0 = (y_train == 0).sum()
n_clase1 = (y_train == 1).sum()

w0 = n_total / (2 * n_clase0)   # clase mayoritaria → peso bajo
w1 = n_total / (2 * n_clase1)   # clase minoritaria → peso alto

print(f"\nPesos calculados:")
print(f"  Clase 0 (mayoritaria): {w0:.3f}")
print(f"  Clase 1 (minoritaria): {w1:.3f}  <= pesa {w1/w0:.0f}× más")

# PYTORCH
print("\n" + ""*45)
print("PYTORCH — CrossEntropyLoss con weight")
print(""*45)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_te = torch.tensor(X_test,  dtype=torch.float32)
y_te = torch.tensor(y_test,  dtype=torch.long)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# La clave: pasar el tensor de pesos a CrossEntropyLoss
# El loss de cada ejemplo se multiplica por el peso de su clase
pesos    = torch.tensor([w0, w1], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=pesos)

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
    rec   = tp / (y_te == 1).sum().item()
    prec  = tp / (preds == 1).sum().item() if (preds == 1).sum() > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

print(f"Accuracy  : {acc:.4f}")
print(f"Recall    : {rec:.4f}  <= detectó {rec:.0%} de clase 1")
print(f"Precision : {prec:.4f}")
print(f"F1-Score  : {f1:.4f}")

# TENSORFLOW
print("\n" + ""*45)
print("TENSORFLOW — class_weight en model.fit()")
print(""*45)

import tensorflow as tf

model_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(2,  activation="softmax"),
])

model_tf.compile(
    optimizer = "adam",
    loss      = "sparse_categorical_crossentropy",
    metrics   = [
        "accuracy",
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
    ]
)

# La clave: diccionario {clase: peso} en model.fit()
class_weight = {0: w0, 1: w1}

model_tf.fit(
    X_train, y_train,
    epochs       = 30,
    batch_size   = 32,
    class_weight = class_weight,   # <= una sola línea extra
    verbose      = 0
)

loss, acc_tf, rec_tf, prec_tf = model_tf.evaluate(X_test, y_test, verbose=0)
f1_tf = 2 * prec_tf * rec_tf / (prec_tf + rec_tf) if (prec_tf + rec_tf) > 0 else 0

print(f"Accuracy  : {acc_tf:.4f}")
print(f"Recall    : {rec_tf:.4f}  <= detectó {rec_tf:.0%} de clase 1")
print(f"Precision : {prec_tf:.4f}")
print(f"F1-Score  : {f1_tf:.4f}")

# COMPARATIVA: CON Y SIN class weight
print("\n" + ""*45)
print("COMPARATIVA (referencia esperada)")
print(""*45)
print(f"{'Métrica':<12} {'Sin weight':>12} {'Con weight':>12}")
print(""*38)
print(f"{'Recall':<12} {'~0.10':>12} {'~0.80+':>12}  <= gran diferencia")
print(f"{'F1-Score':<12} {'~0.15':>12} {'~0.75+':>12}")
print(f"{'Accuracy':<12} {'~0.95':>12} {'~0.88+':>12}  <= baja un poco (normal)")
print("""
Nota: sin class_weight el modelo aprende a predecir
siempre clase 0 (95% acc, 0% recall clase 1).
Con class_weight sacrifica algo de accuracy global
pero detecta correctamente la clase minoritaria.
""")

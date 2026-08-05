"""
Focal Loss reduce el peso de ejemplos fáciles
(clase mayoritaria bien clasificada) y fuerza al
modelo a enfocarse en los ejemplos difíciles.

FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

  p_t : probabilidad predicha para la clase correcta
  γ   : focusing parameter (γ=0 => CrossEntropy normal)
  α_t : balance entre clases (≈ class weight)

REquire: torch tensorflow scikit-learn
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

print("Dataset desequilibrado (sin modificar):")
print(f"  Clase 0: {(y_train==0).sum()}  |  Clase 1: {(y_train==1).sum()}")
print(f"\nConfiguración Focal Loss:")
print(f"  γ (gamma) = 2.0  => ejemplos con p=0.9 pesan 100× menos")
print(f"  α (alpha) = 0.25 => balance entre clases")

# PYTORCH
print("\n" + ""*45)
print("PYTORCH — Focal Loss como nn.Module")
print(""*45)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

#  Implementación de Focal Loss 
class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multiclase.

    Para cada ejemplo en el batch:
      1. Calcula cross-entropy normal => ce_loss
      2. Convierte a probabilidad => p_t = exp(-ce_loss)
      3. Aplica factor de enfoque => (1 - p_t)^gamma
         · Si p_t ≈ 1 (ejemplo fácil): factor ≈ 0  => se ignora
         · Si p_t ≈ 0 (ejemplo difícil): factor ≈ 1 => peso completo
      4. Multiplica por alpha para balancear clases
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        # logits  : (B, C) — salida cruda del modelo
        # targets : (B,)   — etiquetas enteras

        # Cross-entropy por ejemplo (sin reducción)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Probabilidad de la clase correcta
        p_t = torch.exp(-ce_loss)

        # Factor de enfoque: reduce peso de ejemplos fáciles
        focal_factor = (1 - p_t) ** self.gamma

        # Alpha por clase
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha),
            torch.tensor(1 - self.alpha)
        )

        # Pérdida final
        loss = alpha_t * focal_factor * ce_loss
        return loss.mean()


#  Datos 
X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_te = torch.tensor(X_test,  dtype=torch.float32)
y_te = torch.tensor(y_test,  dtype=torch.long)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

#  Modelo 
model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 2)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = FocalLoss(gamma=2.0, alpha=0.25)   # ← Focal Loss

#  Entrenamiento 
for epoch in range(30):
    model.train()
    for X_b, y_b in loader:
        optimizer.zero_grad()
        criterion(model(X_b), y_b).backward()
        optimizer.step()

#  Evaluación 
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

print(f"Accuracy  : {acc:.4f}")
print(f"Recall    : {rec:.4f}  ← detectó {rec:.0%} de clase 1")
print(f"Precision : {prec:.4f}")
print(f"F1-Score  : {f1:.4f}")


# TENSORFLOW

print("\n" + ""*45)
print("TENSORFLOW — Focal Loss como keras.losses.Loss")
print(""*45)

import tensorflow as tf

#  Implementación de Focal Loss 
class FocalLossTF(tf.keras.losses.Loss):
    """
    Focal Loss para TensorFlow/Keras.
    Hereda de tf.keras.losses.Loss — se integra igual
    que cualquier pérdida built-in en model.compile().
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # y_true: etiquetas 0/1  (B,)
        # y_pred: probabilidades (B, 2) — salida del softmax

        # Probabilidad de la clase correcta para cada ejemplo
        y_true_int = tf.cast(y_true, tf.int32)
        indices    = tf.stack([
            tf.range(tf.shape(y_true_int)[0]),
            y_true_int
        ], axis=1)
        p_t = tf.gather_nd(y_pred, indices)

        # Cross-entropy por ejemplo
        ce_loss = -tf.math.log(tf.clip_by_value(p_t, 1e-7, 1.0))

        # Factor de enfoque
        focal_factor = (1.0 - p_t) ** self.gamma

        # Alpha por clase
        alpha_t = tf.where(
            tf.equal(y_true, 1),
            tf.ones_like(p_t) * self.alpha,
            tf.ones_like(p_t) * (1 - self.alpha)
        )

        loss = alpha_t * focal_factor * ce_loss
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


#  Modelo 
model_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2,  activation="softmax"),
])

model_tf.compile(
    optimizer = tf.keras.optimizers.Adam(1e-3),
    loss      = FocalLossTF(gamma=2.0, alpha=0.25),   # ← Focal Loss
    metrics   = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
    ]
)

model_tf.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

results = model_tf.evaluate(X_test, y_test, verbose=0)
_, acc_tf, rec_tf, prec_tf = results
f1_tf = 2*prec_tf*rec_tf/(prec_tf+rec_tf) if (prec_tf+rec_tf) > 0 else 0

print(f"Accuracy  : {acc_tf:.4f}")
print(f"Recall    : {rec_tf:.4f}  ← detectó {rec_tf:.0%} de clase 1")
print(f"Precision : {prec_tf:.4f}")
print(f"F1-Score  : {f1_tf:.4f}")


# EFECTO DEL PARÁMETRO GAMMA
print("\n" + ""*45)
print("Efecto del factor de enfoque (1 - p_t)^γ")
print(""*45)
print(f"{'p_t (prob correcta)':>22} {'γ=0':>8} {'γ=1':>8} {'γ=2':>8} {'γ=5':>8}")
print(""*52)
for p in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
    factores = [f"{(1-p)**g:.4f}" for g in [0, 1, 2, 5]]
    print(f"  p_t = {p:.2f}  {factores[0]:>8} {factores[1]:>8} {factores[2]:>8} {factores[3]:>8}")
print("""
Lectura:
  p_t = 0.10 (ejemplo difícil) => factor alto => modelo aprende de él
  p_t = 0.95 (ejemplo fácil)   => factor bajo => contribuye poco al gradiente
  γ = 0 => Focal Loss = CrossEntropy normal
  γ = 2 => estándar (paper original RetinaNet)
""")

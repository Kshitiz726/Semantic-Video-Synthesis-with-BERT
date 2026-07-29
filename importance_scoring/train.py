import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from scipy.stats import spearmanr

from visual_model import VisualImportanceModel
from visual_dataset import VisualImportanceDataset

# =========================
# PATHS
# =========================

VISUAL_PATH = "features/visual.pt"
LABEL_PATH = "tvsum_segment_labels.json"

# =========================
# HYPERPARAMS
# =========================

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
VAL_SPLIT = 0.2   # 80/20 split

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATA
# =========================

dataset = VisualImportanceDataset(VISUAL_PATH, LABEL_PATH)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train segments: {len(train_ds)} | Val segments: {len(val_ds)}")

# =========================
# MODEL
# =========================

model = VisualImportanceModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAINING LOGS
# =========================

train_losses = []
val_losses = []
val_spearmans = []

# =========================
# TRAIN LOOP
# =========================

for epoch in range(EPOCHS):

    # -------- TRAIN --------
    model.train()
    train_loss = 0.0

    for v, y in train_loader:
        v, y = v.to(device), y.to(device)

        pred = model(v)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # -------- VALIDATE --------
    model.eval()
    val_loss = 0.0
    preds, gts = [], []

    with torch.no_grad():
        for v, y in val_loader:
            v, y = v.to(device), y.to(device)

            pred = model(v)
            loss = criterion(pred, y)
            val_loss += loss.item()

            preds.extend(pred.cpu().numpy().flatten())
            gts.extend(y.cpu().numpy().flatten())

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    spearman = spearmanr(preds, gts)[0]
    val_spearmans.append(spearman)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Spearman: {spearman:.3f}"
    )

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "visual_importance_model.pt")
print("✅ Visual model saved")

# =========================
# SAVE TRAINING LOGS
# =========================

with open("training_logs.json", "w") as f:
    json.dump(
        {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_spearman": val_spearmans
        },
        f,
        indent=2
    )

print("📊 training_logs.json saved")

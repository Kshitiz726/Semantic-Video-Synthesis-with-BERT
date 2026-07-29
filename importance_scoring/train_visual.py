import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
from pathlib import Path

from visual_model import VisualImportanceModel
from visual_dataset import VisualImportanceDataset


VISUAL_PATH = "features/visual.pt"
LABEL_PATH = "tvsum_segment_labels.json"

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4  # L2 regularization
DROPOUT = 0.4
PATIENCE = 5  # Early stopping patience

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Training on: {device}")

# Create dataset and split into train/validation (80/20)
dataset = VisualImportanceDataset(VISUAL_PATH, LABEL_PATH)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"📊 Dataset split: {train_size} train, {val_size} validation")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model with dropout
model = VisualImportanceModel(dropout=DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

print(f"\n🚀 Starting training for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    # ========== Training Phase ==========
    model.train()
    train_loss = 0
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
    
    # ========== Validation Phase ==========
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for v, y in val_loader:
            v, y = v.to(device), y.to(device)
            pred = model(v)
            loss = criterion(pred, y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Print progress
    print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}", end="")
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "visual_importance_model.pt")
        print(f" ✅ Best model saved!")
    else:
        patience_counter += 1
        print(f" ⏳ No improvement ({patience_counter}/{PATIENCE})")
        
    if patience_counter >= PATIENCE:
        print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
        break

print("\n✅ Training completed!")
print(f"📈 Best validation loss: {best_val_loss:.5f}")

# Save training history
history = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "best_val_loss": best_val_loss,
    "epochs_trained": len(train_losses)
}

with open("training_history.json", "w") as f:
    json.dump(history, f, indent=2)

# Generate loss curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', linewidth=2, color='#2196F3')
plt.plot(val_losses, label='Validation Loss', linewidth=2, color='#FF9800')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('Training and Validation Loss (Improved Model)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('improved_loss_curve.png', dpi=150)


with open("training_history.json", "w") as f:
    json.dump(
        {
            "train_loss": train_losses,
            "val_loss": val_losses
        },
        f,
        indent=2
    )



print("📊 Loss curve saved as 'improved_loss_curve.png'")

"""
Test script to verify the model can be loaded correctly
"""
import torch
from visual_model import VisualImportanceModel

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = VisualImportanceModel().to(device)
model.load_state_dict(torch.load("visual_importance_model.pt", map_location=device))
model.eval()

print("✅ Model loaded successfully!")
print(f"Model architecture:")
print(model)

# Test with dummy input
dummy_input = torch.randn(1, 512).to(device)
with torch.no_grad():
    output = model(dummy_input)
    
print(f"\n✅ Test inference successful!")
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output value: {output.item():.4f}")

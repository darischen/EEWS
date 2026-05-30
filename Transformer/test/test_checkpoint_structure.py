"""Check checkpoint structure"""
import torch

checkpoint = torch.load('checkpoints/best_transformer.pth', map_location='cpu')
print("Checkpoint keys:", checkpoint.keys())
print("\nModel state dict keys (first 10):")
for i, key in enumerate(list(checkpoint.keys())[:10]):
    print(f"  {key}")

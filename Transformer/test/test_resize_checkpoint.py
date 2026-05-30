"""
Test script to resize checkpoint from 5D input to 6D input (add sentiment dimension).
Loads best_transformer.pth, resizes input_projection layer, saves as checkpoint_6d.pth
"""
import sys
import os
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.transformer import DeepStockTransformer

def resize_checkpoint_for_sentiment(old_checkpoint_path, new_checkpoint_path):
    """
    Load 5D checkpoint, resize input_projection to 6D, save new checkpoint.

    Strategy:
    - Load old weights (5 -> 128)
    - Create new layer (6 -> 128)
    - Copy old 5 rows, initialize row 5 (sentiment) with small random values
    """
    print(f"Loading checkpoint from {old_checkpoint_path}...")
    checkpoint_orig = torch.load(old_checkpoint_path, map_location='cpu')

    # Make a copy for 5D verification
    checkpoint_5d = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in checkpoint_orig.items()}

    # Make a copy for 6D modification
    checkpoint_6d = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in checkpoint_orig.items()}

    # Extract the input_projection weights (checkpoint is state dict directly)
    old_weight = checkpoint_6d['input_projection.weight']  # (128, 5)
    old_bias = checkpoint_6d['input_projection.bias']      # (128,)

    print(f"Old input_projection.weight shape: {old_weight.shape}")
    print(f"Old input_projection.bias shape: {old_bias.shape}")

    # Create new weight matrix (128, 6)
    new_weight = torch.zeros(old_weight.size(0), old_weight.size(1) + 1)

    # Copy existing 5 columns
    new_weight[:, :5] = old_weight

    # Initialize sentiment column (6th) with small random values
    # Using same scale as Xavier uniform initialization
    new_weight[:, 5].uniform_(-0.1, 0.1)

    print(f"New input_projection.weight shape: {new_weight.shape}")

    # Update checkpoint for 6D
    checkpoint_6d['input_projection.weight'] = new_weight
    # Bias stays the same (output dimension unchanged)

    print(f"Saving resized checkpoint to {new_checkpoint_path}...")
    torch.save(checkpoint_6d, new_checkpoint_path)
    print("[OK] Checkpoint resized and saved!")

    # Verify by loading both models
    print("\n--- Verification ---")
    model_5d = DeepStockTransformer(input_dim=5)
    model_5d.load_state_dict(checkpoint_5d)
    print("[OK] Original 5D checkpoint loads successfully")

    # Load 6D model with resized checkpoint
    model_6d = DeepStockTransformer(input_dim=6)
    model_6d.load_state_dict(checkpoint_6d)
    print("[OK] New 6D model loads resized checkpoint successfully!")

    # Test forward pass
    x_6d = torch.randn(2, 60, 6)  # batch=2, seq_len=60, 6 features
    with torch.no_grad():
        output = model_6d(x_6d)
    print(f"[OK] Forward pass successful!")
    print(f"  Output shapes: 1day={output['1day'].shape}, 5day={output['5day'].shape}, " +
          f"20day={output['20day'].shape}, volatility={output['volatility'].shape}")

if __name__ == '__main__':
    resize_checkpoint_for_sentiment(
        'checkpoints/best_transformer.pth',
        'checkpoints/best_transformer_6d.pth'
    )

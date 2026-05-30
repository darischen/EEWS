"""
Test script to resize checkpoint from 5D input to 6D input (add sentiment dimension).
Loads best_transformer.pth, resizes input_projection layer, saves as checkpoint_6d.pth
"""
import sys
import torch
import torch.nn as nn
sys.path.insert(0, 'src')

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
    checkpoint = torch.load(old_checkpoint_path, map_location='cpu')

    # Extract the input_projection weights
    old_weight = checkpoint['model_state_dict']['input_projection.weight']  # (128, 5)
    old_bias = checkpoint['model_state_dict']['input_projection.bias']      # (128,)

    print(f"Old input_projection.weight shape: {old_weight.shape}")
    print(f"Old input_projection.bias shape: {old_bias.shape}")

    # Create new weight matrix (128, 6)
    new_weight = torch.zeros(old_weight.size(0), old_weight.size(0) + 1)

    # Copy existing 5 columns
    new_weight[:, :5] = old_weight

    # Initialize sentiment column (6th) with small random values
    # Using same scale as Xavier uniform initialization
    new_weight[:, 5].uniform_(-0.1, 0.1)

    print(f"New input_projection.weight shape: {new_weight.shape}")

    # Update checkpoint
    checkpoint['model_state_dict']['input_projection.weight'] = new_weight
    # Bias stays the same (output dimension unchanged)

    # Update config
    if 'config' in checkpoint:
        checkpoint['config']['input_dim'] = 6

    print(f"Saving resized checkpoint to {new_checkpoint_path}...")
    torch.save(checkpoint, new_checkpoint_path)
    print("✓ Checkpoint resized and saved!")

    # Verify by loading both models
    print("\n--- Verification ---")
    model_5d = DeepStockTransformer(input_dim=5)
    model_5d.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Old model (5D) loads successfully")

    # Try loading with 6D
    model_6d = DeepStockTransformer(input_dim=6)
    model_6d.load_state_dict(checkpoint['model_state_dict'])
    print("✓ New model (6D) loads resized checkpoint successfully!")

    # Test forward pass
    x_6d = torch.randn(2, 60, 6)  # batch=2, seq_len=60, 6 features
    with torch.no_grad():
        output = model_6d(x_6d)
    print(f"✓ Forward pass successful!")
    print(f"  Output shapes: 1day={output['1day'].shape}, 5day={output['5day'].shape}, " +
          f"20day={output['20day'].shape}, volatility={output['volatility'].shape}")

if __name__ == '__main__':
    resize_checkpoint_for_sentiment(
        'checkpoints/best_transformer.pth',
        'checkpoints/best_transformer_6d.pth'
    )

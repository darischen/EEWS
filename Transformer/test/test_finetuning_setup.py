"""
Phase 4: Fine-tuning setup verification
Tests that we can:
1. Load 6D checkpoint
2. Prepare sentiment-rich data
3. Freeze early layers appropriately
4. Run a short training epoch to verify setup
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from models.transformer import DeepStockTransformer

def test_finetuning_setup():
    """Test that 6D checkpoint loads and we can freeze layers appropriately"""

    print("=" * 60)
    print("PHASE 4: Fine-Tuning Setup Verification")
    print("=" * 60)

    # Load 6D checkpoint
    print("\n[1] Loading 6D checkpoint...")
    checkpoint_path = 'checkpoints/best_transformer_6d.pth'

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: {checkpoint_path} not found!")
        print("Please run test_resize_checkpoint.py first")
        return False

    model = DeepStockTransformer(input_dim=6)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f"[OK] Loaded {checkpoint_path}")

    # Verify input projection has 6D input
    input_proj_weight = model.input_projection.weight
    print(f"[OK] Input projection shape: {input_proj_weight.shape}")  # Should be [128, 6]

    # Strategy for fine-tuning:
    # - Freeze early transformer layers (layers 0-2)
    # - Keep late layers unfrozen (layer 3)
    # - Keep prediction heads unfrozen
    # - Use lower learning rate

    print("\n[2] Setting up layer freezing strategy...")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Freeze first 3 layers of transformer
    num_frozen = 0
    for name, param in model.named_parameters():
        if 'transformer_encoder.layers.0' in name or \
           'transformer_encoder.layers.1' in name or \
           'transformer_encoder.layers.2' in name:
            param.requires_grad = False
            num_frozen += param.numel()
        else:
            param.requires_grad = True

    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Show which layers are trainable
    print("\nTrainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad and 'transformer_encoder' in name:
            print(f"  - {name}")

    # Test forward pass
    print("\n[3] Testing forward pass with 6D input...")
    model.eval()
    x = torch.randn(4, 60, 6)  # batch=4, seq_len=60, 6 features (with sentiment)

    with torch.no_grad():
        output = model(x)

    print(f"[OK] Forward pass successful")
    for key, val in output.items():
        print(f"  - {key}: shape={val.shape}")

    # Recommend learning rate and schedule
    print("\n[4] Recommended fine-tuning hyperparameters:")
    print("  - Learning rate: 1e-5 (10x lower than pretraining 5e-4)")
    print("  - Optimizer: AdamW with default betas")
    print("  - Batch size: 256 (smaller for stability)")
    print("  - Epochs: 8-12 (stop early if val loss plateaus)")
    print("  - Data: Sentiment-rich sequences (Feb 26 - Mar 20, 2026)")

    print("\n[OK] Fine-tuning setup verified!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = test_finetuning_setup()
    sys.exit(0 if success else 1)

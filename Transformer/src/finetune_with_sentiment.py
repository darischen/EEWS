"""
Phase 4: Fine-tune transformer on sentiment-rich data (Feb 26 - Mar 20, 2026)

Usage:
  python -m src.finetune_with_sentiment

This script:
1. Loads 6D checkpoint (with resized input projection)
2. Freezes early layers (0-2), keeps late layers (3) unfrozen
3. Loads sentiment-rich sequences from data/sentiment/sequences_for_finetuning/
4. Fine-tunes on Feb-Mar 2026 data with low learning rate
5. Saves best checkpoint to checkpoints/sentiment_finetuned.pth
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, 'src')
from models.transformer import DeepStockTransformer
from training.train import validate

def load_sentiment_sequences():
    """Load 6D sequences with sentiment from data/sentiment/sequences_for_finetuning/"""
    seq_dir = Path('data/sentiment/sequences_for_finetuning')

    if not seq_dir.exists():
        raise FileNotFoundError(
            f"Sentiment sequence directory not found: {seq_dir}\n"
            "Run: python -m src.sentiment.create_sentiment_sequences"
        )

    X = np.load(seq_dir / 'X.npy')
    y_1d = np.load(seq_dir / 'y_1d.npy')
    y_5d = np.load(seq_dir / 'y_5d.npy')
    y_20d = np.load(seq_dir / 'y_20d.npy')
    y_vol = np.load(seq_dir / 'y_vol.npy')

    print(f"Loaded sentiment sequences:")
    print(f"  X shape: {X.shape} (batch, seq_len, features)")
    print(f"  y_1d shape: {y_1d.shape}")
    print(f"  Total sequences: {len(X)}")

    return X, y_1d, y_5d, y_20d, y_vol

def finetune():
    """Fine-tune model on sentiment-rich data"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print("\n[1] Loading 6D checkpoint...")
    checkpoint_path = 'checkpoints/best_transformer_6d.pth'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run: python test/test_resize_checkpoint.py"
        )

    model = DeepStockTransformer(input_dim=6)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    print(f"[OK] Loaded checkpoint with 6D input projection")

    # Freeze early layers
    print("\n[2] Freezing early transformer layers...")
    for name, param in model.named_parameters():
        if 'transformer_encoder.layers.0' in name or \
           'transformer_encoder.layers.1' in name or \
           'transformer_encoder.layers.2' in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Trainable params: {trainable_params:,} / {total_params:,}")

    # Load sentiment sequences
    print("\n[3] Loading sentiment-rich sequences...")
    X, y_1d, y_5d, y_20d, y_vol = load_sentiment_sequences()

    # Create dataset
    X_tensor = torch.from_numpy(X).float()
    y_1d_tensor = torch.from_numpy(y_1d).float()
    y_5d_tensor = torch.from_numpy(y_5d).float()
    y_20d_tensor = torch.from_numpy(y_20d).float()
    y_vol_tensor = torch.from_numpy(y_vol).float()

    # 80/20 train/val split
    n_train = int(0.8 * len(X))
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = TensorDataset(
        X_tensor[train_idx],
        y_1d_tensor[train_idx],
        y_5d_tensor[train_idx],
        y_20d_tensor[train_idx],
        y_vol_tensor[train_idx]
    )

    val_dataset = TensorDataset(
        X_tensor[val_idx],
        y_1d_tensor[val_idx],
        y_5d_tensor[val_idx],
        y_20d_tensor[val_idx],
        y_vol_tensor[val_idx]
    )

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[OK] Train: {len(train_idx)} sequences, Val: {len(val_idx)} sequences")
    print(f"    Batch size: {batch_size}")

    # Setup training
    print("\n[4] Setting up fine-tuning...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,  # 10x lower than pretraining
        weight_decay=1e-5
    )

    loss_fn = nn.MSELoss()
    epochs = 12
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Learning rate: 1e-5")
    print(f"Optimizer: AdamW")
    print(f"Epochs: {epochs}")
    print(f"Patience: {patience}")

    # Fine-tuning loop
    print("\n[5] Starting fine-tuning...")
    print("=" * 70)

    history = {
        'train_loss': [],
        'train_1d': [], 'train_5d': [], 'train_20d': [], 'train_vol': [],
        'val_loss': [],
        'val_1d': [], 'val_5d': [], 'val_20d': [], 'val_vol': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        task_losses = {f'1d': 0, f'5d': 0, f'20d': 0, 'vol': 0}

        for batch_idx, (x, y1d, y5d, y20d, yvol) in enumerate(train_loader):
            x = x.to(device)
            y1d = y1d.to(device)
            y5d = y5d.to(device)
            y20d = y20d.to(device)
            yvol = yvol.to(device)

            optimizer.zero_grad()

            output = model(x)
            loss_1d = loss_fn(output['1day'], y1d)
            loss_5d = loss_fn(output['5day'], y5d)
            loss_20d = loss_fn(output['20day'], y20d)
            loss_vol = loss_fn(output['volatility'], yvol)

            total_loss = loss_1d + loss_5d + loss_20d + loss_vol
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += total_loss.item()
            task_losses['1d'] += loss_1d.item()
            task_losses['5d'] += loss_5d.item()
            task_losses['20d'] += loss_20d.item()
            task_losses['vol'] += loss_vol.item()

        train_loss /= len(train_loader)
        for k in task_losses:
            task_losses[k] /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_task_losses = {f'1d': 0, f'5d': 0, f'20d': 0, 'vol': 0}

        with torch.no_grad():
            for x, y1d, y5d, y20d, yvol in val_loader:
                x = x.to(device)
                y1d = y1d.to(device)
                y5d = y5d.to(device)
                y20d = y20d.to(device)
                yvol = yvol.to(device)

                output = model(x)
                loss_1d = loss_fn(output['1day'], y1d)
                loss_5d = loss_fn(output['5day'], y5d)
                loss_20d = loss_fn(output['20day'], y20d)
                loss_vol = loss_fn(output['volatility'], yvol)

                total_loss = loss_1d + loss_5d + loss_20d + loss_vol

                val_loss += total_loss.item()
                val_task_losses['1d'] += loss_1d.item()
                val_task_losses['5d'] += loss_5d.item()
                val_task_losses['20d'] += loss_20d.item()
                val_task_losses['vol'] += loss_vol.item()

        val_loss /= len(val_loader)
        for k in val_task_losses:
            val_task_losses[k] /= len(val_loader)

        # Log
        print(f"Epoch {epoch+1:2d}/{epochs} | Train={train_loss:.4f} " +
              f"(1d={task_losses['1d']:.4f} 5d={task_losses['5d']:.4f} " +
              f"20d={task_losses['20d']:.4f} vol={task_losses['vol']:.4f}) | " +
              f"Val={val_loss:.4f} " +
              f"(1d={val_task_losses['1d']:.4f} 5d={val_task_losses['5d']:.4f} " +
              f"20d={val_task_losses['20d']:.4f} vol={val_task_losses['vol']:.4f})")

        history['train_loss'].append(train_loss)
        history['train_1d'].append(task_losses['1d'])
        history['train_5d'].append(task_losses['5d'])
        history['train_20d'].append(task_losses['20d'])
        history['train_vol'].append(task_losses['vol'])
        history['val_loss'].append(val_loss)
        history['val_1d'].append(val_task_losses['1d'])
        history['val_5d'].append(val_task_losses['5d'])
        history['val_20d'].append(val_task_losses['20d'])
        history['val_vol'].append(val_task_losses['vol'])

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/sentiment_finetuned.pth')
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("=" * 70)
    print("\n[OK] Fine-tuning complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved: checkpoints/sentiment_finetuned.pth")

if __name__ == '__main__':
    finetune()

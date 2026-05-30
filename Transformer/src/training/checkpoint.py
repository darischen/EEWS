import torch
import os
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save full training checkpoint (model + optimizer + metadata)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path} (epoch {epoch}, val_loss {val_loss:.6f})")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """Load training checkpoint and restore model + optimizer state"""
    if not os.path.exists(path):
        logger.warning(f"No checkpoint found at {path}")
        return model, optimizer, 0, float('inf')

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))

    logger.info(f"Checkpoint loaded from {path} (epoch {epoch}, val_loss {val_loss:.6f})")
    return model, optimizer, epoch, val_loss

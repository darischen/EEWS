import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging
import gc

logger = logging.getLogger(__name__)

def get_lr_scheduler(optimizer, epochs, warmup_epochs=5):
    """Cosine annealing with linear warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, train_loader, optimizer, loss_fn, device, scaler):
    """Train single epoch"""
    model.train()
    total_loss = 0
    task_losses = {'1day': 0, '5day': 0, '20day': 0, 'volatility': 0}

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_x, targets in pbar:
        batch_x = batch_x.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            predictions = model(batch_x)
            loss, task_loss_dict = loss_fn(predictions, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for task, task_loss in task_loss_dict.items():
            task_losses[task] += task_loss
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(train_loader)
    avg_task_losses = {k: v / len(train_loader) for k, v in task_losses.items()}
    return avg_loss, avg_task_losses

def validate(model, val_loader, loss_fn, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    task_losses = {'1day': 0, '5day': 0, '20day': 0, 'volatility': 0}

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for batch_x, targets in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            with autocast('cuda', dtype=torch.float16):
                predictions = model(batch_x)
                loss, task_loss_dict = loss_fn(predictions, targets)

            total_loss += loss.item()
            for task, task_loss in task_loss_dict.items():
                task_losses[task] += task_loss
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(val_loader)
    avg_task_losses = {k: v / len(val_loader) for k, v in task_losses.items()}
    return avg_loss, avg_task_losses

def train_model(model, train_loader, val_loader, device, epochs=50, lr=5e-4,
                loss_fn=None, checkpoint_path='best_model.pth', patience=15, test_loader=None):
    """Full training loop with early stopping and test evaluation"""

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = get_lr_scheduler(optimizer, epochs, warmup_epochs=5)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    test_losses = [] if test_loader is not None else None

    logger.info(f"Starting training: {epochs} epochs, lr={lr}")

    for epoch in range(epochs):
        train_loss, train_tasks = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_tasks = validate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Evaluate on test set if provided
        if test_loader is not None:
            test_loss, test_tasks = validate(model, test_loader, loss_fn, device)
            test_losses.append(test_loss)
        else:
            test_loss = None
            test_tasks = None

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log overall and per-task losses
        if test_loss is not None:
            logger.info(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Test: {test_loss:.6f} | LR: {current_lr:.6f}")
        else:
            logger.info(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.6f}")

        logger.info(f"         | Train tasks: 1d={train_tasks['1day']:.6f} 5d={train_tasks['5day']:.6f} 20d={train_tasks['20day']:.6f} vol={train_tasks['volatility']:.6f}")
        logger.info(f"         | Val tasks:   1d={val_tasks['1day']:.6f} 5d={val_tasks['5day']:.6f} 20d={val_tasks['20day']:.6f} vol={val_tasks['volatility']:.6f}")
        if test_tasks is not None:
            logger.info(f"         | Test tasks:  1d={test_tasks['1day']:.6f} 5d={test_tasks['5day']:.6f} 20d={test_tasks['20day']:.6f} vol={test_tasks['volatility']:.6f}")

        # Free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"[OK] New best model saved")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.6f}")

    history = {'train': train_losses, 'val': val_losses}
    if test_losses is not None:
        history['test'] = test_losses

    return model, history

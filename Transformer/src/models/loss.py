import torch
import torch.nn as nn

def multi_task_loss(predictions, targets, weights=None):
    """Multi-task loss: 1day + 5day + 20day + volatility"""

    if weights is None:
        weights = {
            '1day': 0.5,
            '5day': 0.25,
            '20day': 0.15,
            'volatility': 0.1
        }

    mse = nn.MSELoss()

    loss_1day = mse(predictions['1day'].squeeze(), targets['1day'].squeeze())
    loss_5day = mse(predictions['5day'].squeeze(), targets['5day'].squeeze())
    loss_20day = mse(predictions['20day'].squeeze(), targets['20day'].squeeze())
    loss_vol = mse(predictions['volatility'].squeeze(), targets['volatility'].squeeze())

    total_loss = (weights['1day'] * loss_1day +
                  weights['5day'] * loss_5day +
                  weights['20day'] * loss_20day +
                  weights['volatility'] * loss_vol)

    return total_loss, {
        '1day': loss_1day.item(),
        '5day': loss_5day.item(),
        '20day': loss_20day.item(),
        'volatility': loss_vol.item()
    }

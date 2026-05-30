"""
Inference utilities for making predictions
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def predict_batch(model, X, device):
    """Single prediction pass (no uncertainty)"""
    model.eval()
    with torch.no_grad():
        output = model(X.to(device))

    return {k: v.cpu() for k, v in output.items()}


def predict_with_uncertainty(model, X, device, num_passes=10):
    """
    Monte Carlo dropout for uncertainty estimation

    Args:
        model: Trained model
        X: Input tensor (batch, seq_len, features)
        device: torch device
        num_passes: Number of forward passes with dropout enabled

    Returns:
        tuple of (predictions_dict, uncertainties_dict)
    """

    model.train()  # Keep dropout enabled
    all_predictions = {'1day': [], '5day': [], '20day': [], 'volatility': []}

    with torch.no_grad():
        for i in range(num_passes):
            output = model(X.to(device))
            for key in all_predictions.keys():
                all_predictions[key].append(output[key].cpu().numpy())

    # Compute mean and std across MC passes
    predictions = {}
    uncertainties = {}

    for key in all_predictions.keys():
        stacked = np.concatenate(all_predictions[key], axis=0)  # (num_passes, batch_size, 1)
        mean = np.mean(stacked, axis=0)  # (batch_size, 1)
        std = np.std(stacked, axis=0)   # (batch_size, 1)

        predictions[key] = torch.from_numpy(mean).float()
        uncertainties[key] = torch.from_numpy(std).float()

    return predictions, uncertainties


def predict_single_ticker(model, X, device, num_passes=10):
    """
    Make prediction for a single ticker

    Args:
        model: Trained transformer model
        X: Input tensor shape (1, seq_len, input_dim) - single sample
        device: torch device (cpu or cuda)
        num_passes: MC dropout passes for uncertainty

    Returns:
        tuple of (predictions_dict, uncertainties_dict)
        where dicts have keys: '1day', '5day', '20day', 'volatility'
        and values are scalars
    """

    predictions, uncertainties = predict_with_uncertainty(
        model, X, device, num_passes=num_passes
    )

    # Extract scalar values from batch dimension
    predictions_scalar = {
        k: v.squeeze()[0] if v.squeeze().dim() > 0 else v.squeeze()
        for k, v in predictions.items()
    }
    uncertainties_scalar = {
        k: v.squeeze()[0] if v.squeeze().dim() > 0 else v.squeeze()
        for k, v in uncertainties.items()
    }

    return predictions_scalar, uncertainties_scalar


def predict_batch_tickers(model, X_dict, device, num_passes=10):
    """
    Make predictions for multiple tickers

    Args:
        model: Trained model
        X_dict: Dict of {ticker: X_tensor} where X_tensor is (1, seq_len, input_dim)
        device: torch device
        num_passes: MC dropout passes

    Returns:
        dict of {ticker: (predictions, uncertainties)}
    """

    results = {}

    for ticker, X in X_dict.items():
        predictions, uncertainties = predict_single_ticker(
            model, X, device, num_passes=num_passes
        )
        results[ticker] = (predictions, uncertainties)

    return results

"""
Quick Phase 5 validation - sample only 10 major tickers to speed up evaluation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from models.transformer import DeepStockTransformer

# Major tickers to sample
SAMPLE_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']

def load_test_data_quick():
    """Load test data from 2016-2026 period, sampling major tickers only"""
    print("Loading test data (sampling 10 major tickers)...")

    ticker_data = {}
    for ticker in SAMPLE_TICKERS:
        # Try stocks folder first, then etfs
        csv_path = Path(f'../data/stocks/{ticker}.csv')
        if not csv_path.exists():
            csv_path = Path(f'../data/etfs/{ticker}.csv')

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=['Date'])
                df = df.sort_values('Date')
                ticker_data[ticker] = df
            except Exception as e:
                print(f"  Skipped {ticker}: {e}")

    if not ticker_data:
        print("ERROR: No ticker data found")
        return None

    # Find global date range
    all_dates = []
    for ticker, data in ticker_data.items():
        if len(data) > 0:
            all_dates.extend(data['Date'].values)

    all_dates = sorted(set(all_dates))
    train_cutoff = all_dates[int(0.70 * len(all_dates))]
    val_cutoff = all_dates[int(0.85 * len(all_dates))]

    print(f"Date split: {len(all_dates)} total dates")
    print(f"  Train: < {pd.Timestamp(train_cutoff).date()}")
    print(f"  Val: < {pd.Timestamp(val_cutoff).date()}")
    print(f"  Test: >= {pd.Timestamp(val_cutoff).date()}")

    # Create test sequences
    X_test, y_1d, y_5d, y_20d, y_vol = [], [], [], [], []

    for ticker, data in ticker_data.items():
        test_data = data[data['Date'] > val_cutoff].copy()
        if len(test_data) < 65:
            continue

        close_col = 'close' if 'close' in test_data.columns else 'Close'
        cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in test_data.columns for c in cols):
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        ohlcv = test_data[cols].values.astype(float)
        seq_len = 60

        for i in range(len(ohlcv) - seq_len):
            X_test.append(ohlcv[i:i+seq_len])
            y_1d.append(test_data.iloc[i+seq_len][close_col])
            if i + seq_len + 4 < len(test_data):
                y_5d.append(test_data.iloc[i+seq_len+4][close_col])
            else:
                y_5d.append(test_data.iloc[-1][close_col])
            if i + seq_len + 19 < len(test_data):
                y_20d.append(test_data.iloc[i+seq_len+19][close_col])
            else:
                y_20d.append(test_data.iloc[-1][close_col])
            if i + seq_len + 4 < len(test_data):
                y_vol.append(test_data.iloc[i+seq_len:i+seq_len+5][close_col].std())
            else:
                y_vol.append(test_data.iloc[-5:][close_col].std())

    if len(X_test) == 0:
        print("ERROR: No test sequences created")
        return None

    X_test = np.array(X_test)
    y_1d = np.array(y_1d)
    y_5d = np.array(y_5d)
    y_20d = np.array(y_20d)
    y_vol = np.array(y_vol)

    print(f"Created test sequences: {X_test.shape}")
    return X_test, y_1d, y_5d, y_20d, y_vol

def evaluate_model(model, X_test, y_1d, y_5d, y_20d, y_vol, model_name, device):
    """Evaluate model on test set"""
    model.eval()
    loss_fn = nn.MSELoss()

    X_tensor = torch.from_numpy(X_test).float().to(device)
    y_1d_tensor = torch.from_numpy(y_1d).float().to(device)
    y_5d_tensor = torch.from_numpy(y_5d).float().to(device)
    y_20d_tensor = torch.from_numpy(y_20d).float().to(device)
    y_vol_tensor = torch.from_numpy(y_vol).float().to(device)

    dataset = TensorDataset(X_tensor, y_1d_tensor, y_5d_tensor, y_20d_tensor, y_vol_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_losses = {'1d': [], '5d': [], '20d': [], 'vol': []}
    total_loss = 0

    with torch.no_grad():
        for x, y1d, y5d, y20d, yvol in loader:
            output = model(x)
            loss_1d = loss_fn(output['1day'], y1d).item()
            loss_5d = loss_fn(output['5day'], y5d).item()
            loss_20d = loss_fn(output['20day'], y20d).item()
            loss_vol = loss_fn(output['volatility'], yvol).item()

            all_losses['1d'].append(loss_1d)
            all_losses['5d'].append(loss_5d)
            all_losses['20d'].append(loss_20d)
            all_losses['vol'].append(loss_vol)
            total_loss += loss_1d + loss_5d + loss_20d + loss_vol

    results = {
        'model': model_name,
        'total_loss': total_loss / len(loader),
        'loss_1d': np.mean(all_losses['1d']),
        'loss_5d': np.mean(all_losses['5d']),
        'loss_20d': np.mean(all_losses['20d']),
        'loss_vol': np.mean(all_losses['vol']),
        'loss_1d_std': np.std(all_losses['1d']),
        'loss_5d_std': np.std(all_losses['5d']),
        'loss_20d_std': np.std(all_losses['20d']),
        'loss_vol_std': np.std(all_losses['vol']),
    }
    return results, all_losses

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load test data
    print("[1] Loading test data...")
    test_data = load_test_data_quick()
    if test_data is None:
        return
    X_test, y_1d, y_5d, y_20d, y_vol = test_data

    # Load models
    print("\n[2] Loading models...")
    model_5d = DeepStockTransformer(input_dim=5)
    checkpoint_5d = torch.load('checkpoints/best_transformer.pth', map_location=device)
    model_5d.load_state_dict(checkpoint_5d)
    model_5d = model_5d.to(device)
    print("[OK] Original 5D model loaded")

    model_6d = DeepStockTransformer(input_dim=6)
    checkpoint_6d = torch.load('checkpoints/sentiment_finetuned.pth', map_location=device)
    model_6d.load_state_dict(checkpoint_6d)
    model_6d = model_6d.to(device)
    print("[OK] Sentiment-finetuned 6D model loaded")

    # Prepare test data
    print("\n[3] Preparing test data...")
    X_test_5d = X_test
    X_test_6d = np.concatenate([X_test, np.zeros((X_test.shape[0], X_test.shape[1], 1))], axis=-1)
    print(f"5D shape: {X_test_5d.shape}")
    print(f"6D shape: {X_test_6d.shape}")

    # Evaluate
    print("\n[4] Evaluating models...")
    print("=" * 80)

    results_5d, losses_5d = evaluate_model(model_5d, X_test_5d, y_1d, y_5d, y_20d, y_vol,
                                           "Original 5D", device)
    print("\nOriginal 5D Model:")
    print(f"  Total Loss:  {results_5d['total_loss']:.6f}")
    print(f"  1-day Loss:  {results_5d['loss_1d']:.6f} ± {results_5d['loss_1d_std']:.6f}")
    print(f"  5-day Loss:  {results_5d['loss_5d']:.6f} ± {results_5d['loss_5d_std']:.6f}")
    print(f"  20-day Loss: {results_5d['loss_20d']:.6f} ± {results_5d['loss_20d_std']:.6f}")
    print(f"  Vol Loss:    {results_5d['loss_vol']:.6f} ± {results_5d['loss_vol_std']:.6f}")

    results_6d, losses_6d = evaluate_model(model_6d, X_test_6d, y_1d, y_5d, y_20d, y_vol,
                                           "Sentiment-Finetuned 6D", device)
    print("\nSentiment-Finetuned 6D Model:")
    print(f"  Total Loss:  {results_6d['total_loss']:.6f}")
    print(f"  1-day Loss:  {results_6d['loss_1d']:.6f} ± {results_6d['loss_1d_std']:.6f}")
    print(f"  5-day Loss:  {results_6d['loss_5d']:.6f} ± {results_6d['loss_5d_std']:.6f}")
    print(f"  20-day Loss: {results_6d['loss_20d']:.6f} ± {results_6d['loss_20d_std']:.6f}")
    print(f"  Vol Loss:    {results_6d['loss_vol']:.6f} ± {results_6d['loss_vol_std']:.6f}")

    # Compute improvements
    print("\n[5] Improvement Analysis:")
    print("=" * 80)

    total_improvement = (results_5d['total_loss'] - results_6d['total_loss']) / results_5d['total_loss'] * 100
    loss_1d_improvement = (results_5d['loss_1d'] - results_6d['loss_1d']) / results_5d['loss_1d'] * 100
    loss_5d_improvement = (results_5d['loss_5d'] - results_6d['loss_5d']) / results_5d['loss_5d'] * 100
    loss_20d_improvement = (results_5d['loss_20d'] - results_6d['loss_20d']) / results_5d['loss_20d'] * 100
    loss_vol_improvement = (results_5d['loss_vol'] - results_6d['loss_vol']) / results_5d['loss_vol'] * 100

    vol_1d_stability = (results_5d['loss_1d_std'] - results_6d['loss_1d_std']) / results_5d['loss_1d_std'] * 100
    vol_5d_stability = (results_5d['loss_5d_std'] - results_6d['loss_5d_std']) / results_5d['loss_5d_std'] * 100
    vol_20d_stability = (results_5d['loss_20d_std'] - results_6d['loss_20d_std']) / results_5d['loss_20d_std'] * 100
    vol_vol_stability = (results_5d['loss_vol_std'] - results_6d['loss_vol_std']) / results_5d['loss_vol_std'] * 100

    print(f"Total Loss Improvement:      {total_improvement:+.2f}%")
    print(f"1-day Loss Improvement:      {loss_1d_improvement:+.2f}%")
    print(f"5-day Loss Improvement:      {loss_5d_improvement:+.2f}%")
    print(f"20-day Loss Improvement:     {loss_20d_improvement:+.2f}%")
    print(f"Volatility Loss Improvement: {loss_vol_improvement:+.2f}%")

    print(f"\nStability Improvement (std dev change):")
    print(f"1-day Stability:   {vol_1d_stability:+.2f}%")
    print(f"5-day Stability:   {vol_5d_stability:+.2f}%")
    print(f"20-day Stability:  {vol_20d_stability:+.2f}%")
    print(f"Vol Stability:     {vol_vol_stability:+.2f}%")

    print(f"\n[6] Conclusion:")
    print("=" * 80)
    if total_improvement > 0:
        print(f"Sentiment fine-tuning IMPROVED performance by {total_improvement:.2f}%")
    else:
        print(f"Sentiment fine-tuning did NOT improve performance ({total_improvement:.2f}%)")

if __name__ == '__main__':
    main()

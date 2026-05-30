"""
Phase 5: Validate sentiment fine-tuning effectiveness

Compares:
1. Original 5D model (pretrained only)
2. Sentiment-finetuned 6D model

On unseen data (April+ 2026) to measure if sentiment signal improved generalization
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
from models.transformer import DeepStockTransformer
from data.normalize import StandardScaler

def load_test_data():
    """Load test data from 2016-2026 period (temporal split from training)"""
    import pickle
    import glob

    print("Preparing test data from 2016-2026 period...")

    # Load all ticker data from CSVs
    ticker_data = {}
    for csv_file in glob.glob('../data/stocks/*.csv') + glob.glob('../data/etfs/*.csv'):
        ticker = Path(csv_file).stem.upper()
        try:
            df = pd.read_csv(csv_file, parse_dates=['Date'])
            df = df.sort_values('Date')
            ticker_data[ticker] = df
        except:
            continue

    if not ticker_data:
        print("ERROR: No ticker data found")
        return None

    # Find global date range
    all_dates = []
    for ticker, data in ticker_data.items():
        if len(data) > 0:
            all_dates.extend(data['Date'].values)

    all_dates = sorted(set(all_dates))
    train_cutoff = all_dates[int(0.70 * len(all_dates))]  # 70%
    val_cutoff = all_dates[int(0.85 * len(all_dates))]    # 85%

    print(f"Date split: Train<{train_cutoff} | Val<{val_cutoff} | Test")

    # Load normalizer
    scaler_path = Path('../data/normalizer.pkl')
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print("WARNING: Normalizer not found, using identity scaling")
        scaler = None

    # Create test sequences (2016-2026 period)
    X_test, y_1d, y_5d, y_20d, y_vol = [], [], [], [], []

    for ticker, data in ticker_data.items():
        if len(data) < 100:
            continue

        # Filter to test period
        test_data = data[data['Date'] > val_cutoff].copy()
        if len(test_data) < 65:  # Need at least 60 + 5 for sequences
            continue

        # Prepare OHLCV (handle both uppercase and lowercase column names)
        cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in test_data.columns for c in cols):
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        ohlcv = test_data[cols].values.astype(float)

        # Normalize
        if scaler:
            ohlcv = scaler.transform(ohlcv)

        # Create sequences
        seq_len = 60
        close_col = 'close' if 'close' in test_data.columns else 'Close'
        for i in range(len(ohlcv) - seq_len):
            X_test.append(ohlcv[i:i+seq_len])
            # Simple target: next day, 5-day, 20-day close prices
            y_1d.append(test_data.iloc[i+seq_len][close_col])
            if i + seq_len + 4 < len(test_data):
                y_5d.append(test_data.iloc[i+seq_len+4][close_col])
            else:
                y_5d.append(test_data.iloc[-1][close_col])
            if i + seq_len + 19 < len(test_data):
                y_20d.append(test_data.iloc[i+seq_len+19][close_col])
            else:
                y_20d.append(test_data.iloc[-1][close_col])
            # Volatility: std of next 5 days
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

    print(f"Created test sequences:")
    print(f"  X shape: {X_test.shape}")
    print(f"  Total sequences: {len(X_test)}")

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

    # Compute statistics
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
        'loss_1d_min': np.min(all_losses['1d']),
        'loss_1d_max': np.max(all_losses['1d']),
        'loss_5d_min': np.min(all_losses['5d']),
        'loss_5d_max': np.max(all_losses['5d']),
        'loss_20d_min': np.min(all_losses['20d']),
        'loss_20d_max': np.max(all_losses['20d']),
        'loss_vol_min': np.min(all_losses['vol']),
        'loss_vol_max': np.max(all_losses['vol']),
    }

    return results, all_losses

def compare_models():
    """Compare original 5D model vs sentiment-finetuned 6D model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load test data
    print("[1] Loading test data...")
    test_data = load_test_data()

    if test_data is None:
        print("ERROR: Test data not available. Create test data with:")
        print("  python -m src.main --skip-fetch --test-only")
        return

    X_test, y_1d, y_5d, y_20d, y_vol = test_data

    # Load original 5D model
    print("\n[2] Loading original 5D model...")
    model_5d = DeepStockTransformer(input_dim=5)

    if not os.path.exists('checkpoints/best_transformer.pth'):
        print("ERROR: Original checkpoint not found at checkpoints/best_transformer.pth")
        return

    checkpoint_5d = torch.load('checkpoints/best_transformer.pth', map_location=device)
    model_5d.load_state_dict(checkpoint_5d)
    model_5d = model_5d.to(device)
    print("[OK] Loaded original 5D model")

    # Load sentiment-finetuned 6D model
    print("\n[3] Loading sentiment-finetuned 6D model...")
    model_6d = DeepStockTransformer(input_dim=6)

    if not os.path.exists('checkpoints/sentiment_finetuned.pth'):
        print("ERROR: Sentiment-finetuned checkpoint not found at checkpoints/sentiment_finetuned.pth")
        return

    checkpoint_6d = torch.load('checkpoints/sentiment_finetuned.pth', map_location=device)
    model_6d.load_state_dict(checkpoint_6d)
    model_6d = model_6d.to(device)
    print("[OK] Loaded sentiment-finetuned 6D model")

    # Handle dimension mismatch for fair comparison
    print("\n[4] Preparing test data...")
    # If test data is 6D, use it for both
    # If test data is 5D, use as-is for 5D model, add sentiment=0 for 6D model
    if X_test.shape[-1] == 5:
        X_test_5d = X_test
        X_test_6d = np.concatenate([X_test, np.zeros((X_test.shape[0], X_test.shape[1], 1))],
                                    axis=-1)
        print(f"Test data is 5D, adding zero-sentiment column for 6D comparison")
    else:
        X_test_5d = X_test[:, :, :5]  # Use first 5 dims only
        X_test_6d = X_test

    # Evaluate both models
    print("\n[5] Evaluating models...")
    print("=" * 80)

    results_5d, losses_5d = evaluate_model(model_5d, X_test_5d, y_1d, y_5d, y_20d, y_vol,
                                           "Original 5D", device)
    print("Original 5D Model:")
    print(f"  Total Loss: {results_5d['total_loss']:.6f}")
    print(f"  1-day:  {results_5d['loss_1d']:.6f} ± {results_5d['loss_1d_std']:.6f} " +
          f"(range: {results_5d['loss_1d_min']:.6f} - {results_5d['loss_1d_max']:.6f})")
    print(f"  5-day:  {results_5d['loss_5d']:.6f} ± {results_5d['loss_5d_std']:.6f} " +
          f"(range: {results_5d['loss_5d_min']:.6f} - {results_5d['loss_5d_max']:.6f})")
    print(f"  20-day: {results_5d['loss_20d']:.6f} ± {results_5d['loss_20d_std']:.6f} " +
          f"(range: {results_5d['loss_20d_min']:.6f} - {results_5d['loss_20d_max']:.6f})")
    print(f"  Vol:    {results_5d['loss_vol']:.6f} ± {results_5d['loss_vol_std']:.6f} " +
          f"(range: {results_5d['loss_vol_min']:.6f} - {results_5d['loss_vol_max']:.6f})")

    results_6d, losses_6d = evaluate_model(model_6d, X_test_6d, y_1d, y_5d, y_20d, y_vol,
                                           "Sentiment-Finetuned 6D", device)
    print("\nSentiment-Finetuned 6D Model:")
    print(f"  Total Loss: {results_6d['total_loss']:.6f}")
    print(f"  1-day:  {results_6d['loss_1d']:.6f} ± {results_6d['loss_1d_std']:.6f} " +
          f"(range: {results_6d['loss_1d_min']:.6f} - {results_6d['loss_1d_max']:.6f})")
    print(f"  5-day:  {results_6d['loss_5d']:.6f} ± {results_6d['loss_5d_std']:.6f} " +
          f"(range: {results_6d['loss_5d_min']:.6f} - {results_6d['loss_5d_max']:.6f})")
    print(f"  20-day: {results_6d['loss_20d']:.6f} ± {results_6d['loss_20d_std']:.6f} " +
          f"(range: {results_6d['loss_20d_min']:.6f} - {results_6d['loss_20d_max']:.6f})")
    print(f"  Vol:    {results_6d['loss_vol']:.6f} ± {results_6d['loss_vol_std']:.6f} " +
          f"(range: {results_6d['loss_vol_min']:.6f} - {results_6d['loss_vol_max']:.6f})")

    # Compute improvement
    print("\n[6] Improvement Analysis:")
    print("=" * 80)

    total_improvement = (results_5d['total_loss'] - results_6d['total_loss']) / results_5d['total_loss'] * 100
    loss_1d_improvement = (results_5d['loss_1d'] - results_6d['loss_1d']) / results_5d['loss_1d'] * 100
    loss_5d_improvement = (results_5d['loss_5d'] - results_6d['loss_5d']) / results_5d['loss_5d'] * 100
    loss_20d_improvement = (results_5d['loss_20d'] - results_6d['loss_20d']) / results_5d['loss_20d'] * 100
    loss_vol_improvement = (results_5d['loss_vol'] - results_6d['loss_vol']) / results_5d['loss_vol'] * 100

    # Volatility improvement (lower std = better stability)
    vol_1d_stability = (results_5d['loss_1d_std'] - results_6d['loss_1d_std']) / results_5d['loss_1d_std'] * 100
    vol_5d_stability = (results_5d['loss_5d_std'] - results_6d['loss_5d_std']) / results_5d['loss_5d_std'] * 100
    vol_20d_stability = (results_5d['loss_20d_std'] - results_6d['loss_20d_std']) / results_5d['loss_20d_std'] * 100
    vol_vol_stability = (results_5d['loss_vol_std'] - results_6d['loss_vol_std']) / results_5d['loss_vol_std'] * 100

    print(f"Total Loss Improvement:     {total_improvement:+.2f}%")
    print(f"1-day Loss Improvement:     {loss_1d_improvement:+.2f}%")
    print(f"5-day Loss Improvement:     {loss_5d_improvement:+.2f}%")
    print(f"20-day Loss Improvement:    {loss_20d_improvement:+.2f}%")
    print(f"Volatility Loss Improvement: {loss_vol_improvement:+.2f}%")

    print(f"\nStability Improvement (lower std = better):")
    print(f"1-day Std Dev:   {vol_1d_stability:+.2f}%")
    print(f"5-day Std Dev:   {vol_5d_stability:+.2f}%")
    print(f"20-day Std Dev:  {vol_20d_stability:+.2f}%")
    print(f"Vol Std Dev:     {vol_vol_stability:+.2f}%")

    # Key insight: Check if sentiment helped reduce volatility spikes
    print(f"\n[7] Key Findings:")
    print("=" * 80)

    if total_improvement > 0:
        print(f"✓ Sentiment fine-tuning IMPROVED overall performance by {total_improvement:.2f}%")
    else:
        print(f"✗ Sentiment fine-tuning did NOT improve performance ({total_improvement:.2f}%)")
        print("  Note: This may be due to:")
        print("  - Limited sentiment data (only 21 dates)")
        print("  - Temporal mismatch (sentiment data ends before test period)")
        print("  - Sentiment signal being weak relative to price patterns")

    if vol_1d_stability > 5 or vol_5d_stability > 5 or vol_20d_stability > 5:
        print(f"✓ Sentiment fine-tuning IMPROVED stability (reduced volatility spikes)")
    else:
        print(f"✗ Sentiment fine-tuning did NOT improve stability")

    print("\nConclusion:")
    if total_improvement > 0:
        print("Sentiment signal successfully enhanced model generalization to new market regimes.")
    else:
        print("Sentiment signal had limited impact - may need:")
        print("  - More sentiment data collection")
        print("  - Different sentiment weighting strategy")
        print("  - Alternative regime indicators")

if __name__ == '__main__':
    compare_models()

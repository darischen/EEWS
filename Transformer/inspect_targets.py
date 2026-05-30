"""Inspect target value distributions to diagnose near-zero loss."""
import numpy as np
import sys
sys.path.insert(0, 'src')

from data.fetch import DataFetcher
from data.normalize import DataNormalizer, create_sequences_per_ticker

# Load data
config = {'data_path': '../data'}
fetcher = DataFetcher(data_path=config['data_path'])
normalizer = DataNormalizer()

# Load all tickers grouped
ticker_data = fetcher.load_tickers_grouped_from_disk()
ticker_name = list(ticker_data.keys())[0]
df = ticker_data[ticker_name]

print(f"\n{'='*60}")
print(f"Inspecting: {ticker_name}")
print(f"{'='*60}\n")

# Get raw OHLCV
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
df = df.ffill().dropna()
available = [col for col in ohlcv_cols if col in df.columns]
ohlcv = df[available].values

print(f"Raw OHLCV shape: {ohlcv.shape}")
print(f"Raw close prices (first 5): {ohlcv[:5, 3]}")
print(f"Raw close prices (last 5): {ohlcv[-5:, 3]}")
print(f"Close price range: [{ohlcv[:, 3].min():.2f}, {ohlcv[:, 3].max():.2f}]")
print(f"Close price std: {np.std(ohlcv[:, 3]):.4f}\n")

# Fit and normalize
all_values = [ohlcv]
normalizer.fit_transform(np.concatenate(all_values))
normalized = normalizer.transform(ohlcv)

print(f"Normalized close prices (first 5): {normalized[:5, 3]}")
print(f"Normalized close prices (last 5): {normalized[-5:, 3]}")
print(f"Normalized close price range: [{normalized[:, 3].min():.6f}, {normalized[:, 3].max():.6f}]")
print(f"Normalized close price std: {np.std(normalized[:, 3]):.6f}\n")

# Create sequences
seq_length = 60
train_tuple, val_tuple, test_tuple = create_sequences_per_ticker({ticker_name: normalized}, seq_length)

X_train, y_1d_train, y_5d_train, y_20d_train, y_vol_train = train_tuple
X_val, y_1d_val, y_5d_val, y_20d_val, y_vol_val = val_tuple
X_test, y_1d_test, y_5d_test, y_20d_test, y_vol_test = test_tuple

print(f"Train sequences: {len(X_train)}")
print(f"Val sequences: {len(X_val)}")
print(f"Test sequences: {len(X_test)}")
print()

# Inspect targets
print("TARGET DISTRIBUTIONS:")
print("-" * 60)
for label, y_1d, y_5d, y_20d, y_vol in [
    ("Train", y_1d_train, y_5d_train, y_20d_train, y_vol_train),
    ("Val", y_1d_val, y_5d_val, y_20d_val, y_vol_val),
    ("Test", y_1d_test, y_5d_test, y_20d_test, y_vol_test),
]:
    print(f"\n{label}:")
    print(f"  1-day:  mean={np.mean(y_1d):.6f}, std={np.std(y_1d):.6f}, range=[{np.min(y_1d):.6f}, {np.max(y_1d):.6f}]")
    print(f"  5-day:  mean={np.mean(y_5d):.6f}, std={np.std(y_5d):.6f}, range=[{np.min(y_5d):.6f}, {np.max(y_5d):.6f}]")
    print(f"  20-day: mean={np.mean(y_20d):.6f}, std={np.std(y_20d):.6f}, range=[{np.min(y_20d):.6f}, {np.max(y_20d):.6f}]")
    print(f"  vol:    mean={np.mean(y_vol):.6f}, std={np.std(y_vol):.6f}, range=[{np.min(y_vol):.6f}, {np.max(y_vol):.6f}]")

# Check for temporal overlap: do later training sequences overlap with val?
print(f"\n{'='*60}")
print("CHECKING FOR SEQUENCE OVERLAP:")
print(f"{'='*60}\n")

last_train_idx = int(0.6 * len(normalized))
first_val_idx = int(0.6 * len(normalized))
print(f"Last training sequence starts at row: {last_train_idx - seq_length}")
print(f"First val sequence starts at row: {first_val_idx}")
print(f"Gap between last train and first val: {first_val_idx - (last_train_idx - seq_length)} rows")
print(f"Sequence length: {seq_length}")
print(f"-> Potential overlap if gap < {seq_length + 20}")

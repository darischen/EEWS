import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
from numba import jit

logger = logging.getLogger(__name__)

class DataNormalizer:
    """Normalize features and create sequences"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, data):
        """Fit scaler on training data, transform"""
        self.fitted = True
        logger.info(f"Fitting scaler on {len(data)} samples...")
        normalized = self.scaler.fit_transform(data)

        logger.info(f"Normalized data shape: {normalized.shape}")
        logger.info(f"Mean: {np.mean(normalized, axis=0)}")
        logger.info(f"Std: {np.std(normalized, axis=0)}")

        return normalized

    def transform(self, data):
        """Transform new data with fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler not fitted yet")
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """Convert back to original scale"""
        if not self.fitted:
            raise ValueError("Scaler not fitted yet")
        return self.scaler.inverse_transform(data)

@jit(nopython=True)
def _create_sequences_numba(data, seq_length):
    """Numba-compiled sequence creation - runs at C speed"""
    n_sequences = len(data) - seq_length - 20

    # Pre-allocate arrays
    X = np.empty((n_sequences, seq_length, data.shape[1]), dtype=np.float32)
    y_1d = np.empty(n_sequences, dtype=np.float32)
    y_5d = np.empty(n_sequences, dtype=np.float32)
    y_20d = np.empty(n_sequences, dtype=np.float32)
    y_vol = np.empty(n_sequences, dtype=np.float32)

    for i in range(n_sequences):
        # Copy sequence
        X[i] = data[i:i+seq_length]

        # Multi-horizon targets
        y_1d[i] = data[i+seq_length, 3]
        y_5d[i] = data[i+seq_length+4, 3]
        y_20d[i] = data[i+seq_length+19, 3]

        # Volatility: compute std of close prices in next 5 days
        vol_values = data[i+seq_length:i+seq_length+5, 3]
        vol_mean = np.mean(vol_values)
        vol_var = 0.0
        for j in range(5):
            vol_var += (vol_values[j] - vol_mean) ** 2
        y_vol[i] = np.sqrt(vol_var / 5.0)

    return X, y_1d, y_5d, y_20d, y_vol


def create_sequences(data, seq_length=60):
    """Create sliding window sequences with multi-horizon targets (Numba-accelerated)"""
    logger.info("Compiling Numba kernel (first run only)...")
    X, y_1d, y_5d, y_20d, y_vol = _create_sequences_numba(data.astype(np.float32), seq_length)
    logger.info(f"Created sequences: X shape {X.shape}")

    return X, y_1d, y_5d, y_20d, y_vol


def create_sequences_per_ticker(ticker_data_dict, seq_length=60):
    """Create sequences within each ticker from already-split (train/val/test) data, then concatenate.

    Args:
        ticker_data_dict: dict[str, np.ndarray] — already-split normalized arrays, one per ticker
        seq_length: length of input sequences

    Returns:
        (X, y_1d, y_5d, y_20d, y_vol) tuple from all tickers concatenated
    """
    min_rows = seq_length + 20  # Need seq_length + 20 future rows for targets

    arrays = {'X': [], 'y_1d': [], 'y_5d': [], 'y_20d': [], 'y_vol': []}
    skipped = 0

    for ticker, data in tqdm(ticker_data_dict.items(), desc="Creating per-ticker sequences", unit="ticker"):
        if len(data) < min_rows + 1:
            skipped += 1
            continue

        X, y_1d, y_5d, y_20d, y_vol = _create_sequences_numba(data.astype(np.float32), seq_length)

        if len(X) == 0:
            skipped += 1
            continue

        # Already split by date in main.py, so use all sequences
        arrays['X'].append(X)
        arrays['y_1d'].append(y_1d)
        arrays['y_5d'].append(y_5d)
        arrays['y_20d'].append(y_20d)
        arrays['y_vol'].append(y_vol)

    if not arrays['X']:
        raise ValueError("No tickers had enough data to create sequences")

    logger.info(f"Processed {len(ticker_data_dict) - skipped} tickers, skipped {skipped} (too short)")

    # Concatenate all tickers
    result = (
        np.concatenate(arrays['X']),
        np.concatenate(arrays['y_1d']),
        np.concatenate(arrays['y_5d']),
        np.concatenate(arrays['y_20d']),
        np.concatenate(arrays['y_vol']),
    )

    logger.info(f"Created sequences: {result[0].shape[0]} total")

    return result

import os
import sys
import torch
import yaml
import logging
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Ensure log and checkpoint directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import custom modules
from .data.fetch import DataFetcher
from .data.loader import EEWSDataLoader
from .data.normalize import DataNormalizer, create_sequences, create_sequences_per_ticker
from .data.chunked_dataset import ChunkedSequenceDataset
from .models.transformer import DeepStockTransformer
from .models.loss import multi_task_loss
from .training.train import train_model
from .inference.predict import predict_with_uncertainty
from .training.checkpoint import load_checkpoint, save_checkpoint

import torch.utils.data as data_utils

def load_config(path='config.yaml'):
    """Load configuration"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def collate_fn(batch):
    """Optimized collate - vectorized numpy operations, return CPU tensors for pin_memory"""
    x, y_1d, y_5d, y_20d, y_vol = zip(*batch)

    # Use np.stack (vectorized) instead of list comprehension
    x_array = np.stack(x, axis=0).astype(np.float32)

    return (torch.from_numpy(x_array), {
        '1day': torch.from_numpy(np.array(y_1d, dtype=np.float32)),
        '5day': torch.from_numpy(np.array(y_5d, dtype=np.float32)),
        '20day': torch.from_numpy(np.array(y_20d, dtype=np.float32)),
        'volatility': torch.from_numpy(np.array(y_vol, dtype=np.float32))
    })

class TrainingPipeline:
    """Manual training pipeline with automatic data updates"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Device: {self.device}")

        self.fetcher = DataFetcher(data_path=config['data_path'])
        self.normalizer = DataNormalizer()
        self.model = None

    def fetch_data(self):
        """Fetch and update ALL tickers from Yahoo Finance, return grouped by ticker"""
        logger.info("Fetching and updating ALL tickers from Yahoo Finance...")

        # Auto-discover all tickers from ../data/stocks and ../data/etfs
        # This also updates the CSV files with latest data
        tickers, success_count, failed = self.fetcher.fetch_and_update_all_tickers()

        logger.info(f"Successfully updated {success_count}/{len(tickers)} tickers")

        # Load ticker data grouped (dict[str, DataFrame]), skip stale tickers
        logger.info("Loading all updated ticker data (grouped by ticker)...")
        ticker_data = self.fetcher.load_tickers_grouped_from_disk(tickers, min_last_trade_days=365)

        logger.info(f"Loaded {len(ticker_data)} tickers")

        return ticker_data

    def prepare_data(self, ticker_data):
        """Clean, split by date, normalize, and create sequences.

        Args:
            ticker_data: dict[str, DataFrame] — raw DataFrames keyed by ticker

        Returns:
            (X_train, y_1d_train, y_5d_train, y_20d_train, y_vol_train,
             X_val, y_1d_val, y_5d_val, y_20d_val, y_vol_val,
             X_test, y_1d_test, y_5d_test, y_20d_test, y_vol_test)
        """
        logger.info("Preparing data (temporal split by date, then per-ticker sequences)...")

        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

        # Step 1: Clean each ticker
        cleaned = {}  # dict[ticker] = (df, ohlcv_array)
        all_values = []  # For fitting global scaler

        for ticker, df in tqdm(ticker_data.items(), desc="Cleaning tickers", unit="ticker"):
            df = df.ffill().dropna()
            if len(df) == 0:
                continue
            available = [col for col in ohlcv_cols if col in df.columns]
            if len(available) < 5:
                continue
            ohlcv = df[available].values
            if len(ohlcv) > 0:
                cleaned[ticker] = (df, ohlcv)
                all_values.append(ohlcv)

        if not cleaned:
            raise ValueError("No tickers had valid OHLCV data after cleaning")

        # Step 2: Fit global scaler on ALL data
        logger.info("Fitting global scaler on all tickers...")
        all_concat = np.concatenate(all_values)
        self.normalizer.fit_transform(all_concat)
        del all_concat, all_values

        # Step 3: Find global date range and split points
        min_date = min(df.index.min() for df, _ in cleaned.values())
        max_date = max(df.index.max() for df, _ in cleaned.values())
        date_range = (max_date - min_date).days

        # Split: 70% train, 15% val, 15% test by calendar time
        train_cutoff = min_date + pd.Timedelta(days=int(date_range * 0.70))
        val_cutoff = min_date + pd.Timedelta(days=int(date_range * 0.85))

        logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
        logger.info(f"Train: {min_date.date()} to {train_cutoff.date()}")
        logger.info(f"Val: {train_cutoff.date()} to {val_cutoff.date()}")
        logger.info(f"Test: {val_cutoff.date()} to {max_date.date()}")

        # Step 4: Split each ticker's data by date, then normalize per-split
        train_dict = {}
        val_dict = {}
        test_dict = {}

        for ticker, (df, ohlcv) in tqdm(cleaned.items(), desc="Splitting tickers by date", unit="ticker"):
            # Split by date indices
            train_mask = df.index <= train_cutoff
            val_mask = (df.index > train_cutoff) & (df.index <= val_cutoff)
            test_mask = df.index > val_cutoff

            if train_mask.sum() > 0:
                train_dict[ticker] = self.normalizer.transform(ohlcv[train_mask])
            if val_mask.sum() > 0:
                val_dict[ticker] = self.normalizer.transform(ohlcv[val_mask])
            if test_mask.sum() > 0:
                test_dict[ticker] = self.normalizer.transform(ohlcv[test_mask])

        del cleaned

        # Step 5: Create sequences per-ticker (now within same time period)
        seq_length = self.config['seq_length']
        logger.info(f"Creating per-ticker sequences (length={seq_length})...")

        train_tuple = create_sequences_per_ticker(train_dict, seq_length) if train_dict else (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        val_tuple = create_sequences_per_ticker(val_dict, seq_length) if val_dict else (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        test_tuple = create_sequences_per_ticker(test_dict, seq_length) if test_dict else (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

        X_train, y_1d_train, y_5d_train, y_20d_train, y_vol_train = train_tuple
        X_val, y_1d_val, y_5d_val, y_20d_val, y_vol_val = val_tuple
        X_test, y_1d_test, y_5d_test, y_20d_test, y_vol_test = test_tuple

        logger.info(f"Train X shape: {X_train.shape}, Val X shape: {X_val.shape}, Test X shape: {X_test.shape}")

        return (X_train, y_1d_train, y_5d_train, y_20d_train, y_vol_train,
                X_val, y_1d_val, y_5d_val, y_20d_val, y_vol_val,
                X_test, y_1d_test, y_5d_test, y_20d_test, y_vol_test)

    def create_dataloaders(self, X_train, y_1d_train, y_5d_train, y_20d_train, y_vol_train,
                               X_val, y_1d_val, y_5d_val, y_20d_val, y_vol_val,
                               X_test, y_1d_test, y_5d_test, y_20d_test, y_vol_test):
        """Create dataloaders from pre-split train/val/test arrays"""

        # Use chunked datasets to keep RAM usage low
        train_dataset = ChunkedSequenceDataset(
            X_train, y_1d_train, y_5d_train, y_20d_train, y_vol_train,
            chunk_dir='data/processed/train',
            chunk_size=10000000  # Larger chunks = fewer seeks, better disk throughput
        )

        val_dataset = ChunkedSequenceDataset(
            X_val, y_1d_val, y_5d_val, y_20d_val, y_vol_val,
            chunk_dir='data/processed/val',
            chunk_size=10000000
        )

        test_dataset = ChunkedSequenceDataset(
            X_test, y_1d_test, y_5d_test, y_20d_test, y_vol_test,
            chunk_dir='data/processed/test',
            chunk_size=10000000
        )

        train_loader = data_utils.DataLoader(
            train_dataset,
            batch_size=int(self.config['batch_size']),
            shuffle=True,  # Shuffle for training quality; cache handles I/O
            collate_fn=collate_fn,
            pin_memory=True,  # True: enables fast pinned memory transfer with non_blocking=True
            num_workers=2,
            persistent_workers=True,  # Keep workers alive between epochs to avoid restart overhead
            prefetch_factor=2
        )

        val_loader = data_utils.DataLoader(
            val_dataset,
            batch_size=int(self.config['batch_size']),
            collate_fn=collate_fn,
            pin_memory=True,  # True: enables fast pinned memory transfer with non_blocking=True
            num_workers=2,
            persistent_workers=True  # Keep workers alive between epochs to avoid restart overhead
        )

        test_loader = data_utils.DataLoader(
            test_dataset,
            batch_size=int(self.config['batch_size']),
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True
        )

        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def train(self, epochs=20, skip_fetch=False):
        """Full training pipeline"""

        logger.info(f"\n{'='*60}")
        logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}\n")

        # Fetch data (automatic) unless skipped
        if skip_fetch:
            logger.info("Skipping data fetch (--skip-fetch flag set)")
            logger.info("Loading existing data from disk (grouped by ticker)...")
            ticker_data = self.fetcher.load_tickers_grouped_from_disk(min_last_trade_days=365)
        else:
            ticker_data = self.fetch_data()

        # Prepare (returns pre-split train/val/test)
        (X_train, y_1d_train, y_5d_train, y_20d_train, y_vol_train,
         X_val, y_1d_val, y_5d_val, y_20d_val, y_vol_val,
         X_test, y_1d_test, y_5d_test, y_20d_test, y_vol_test) = self.prepare_data(ticker_data)

        # Create loaders
        train_loader, val_loader, test_loader = self.create_dataloaders(
            X_train, y_1d_train, y_5d_train, y_20d_train, y_vol_train,
            X_val, y_1d_val, y_5d_val, y_20d_val, y_vol_val,
            X_test, y_1d_test, y_5d_test, y_20d_test, y_vol_test
        )

        # Initialize model
        self.model = DeepStockTransformer(
            input_dim=int(self.config['input_dim']),
            seq_length=int(self.config['seq_length']),
            d_model=int(self.config['d_model']),
            nhead=int(self.config['nhead']),
            num_layers=int(self.config['num_layers']),
            dim_feedforward=int(self.config['dim_feedforward']),
            dropout=float(self.config['dropout'])
        ).to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {param_count:,}\n")

        # Train
        self.model, history = train_model(
            self.model,
            train_loader,
            val_loader,
            self.device,
            epochs=epochs,
            lr=float(self.config['learning_rate']),
            loss_fn=multi_task_loss,
            checkpoint_path=self.config['checkpoint_path'],
            patience=int(self.config['patience']),
            test_loader=test_loader
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Model saved to {self.config['checkpoint_path']}")
        logger.info(f"{'='*60}\n")

        return self.model

    def predict(self, X_test):
        """Generate predictions with uncertainty"""
        if self.model is None:
            checkpoint_path = self.config['checkpoint_path']
            self.model = DeepStockTransformer(
                input_dim=self.config['input_dim'],
                seq_length=self.config['seq_length'],
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                num_layers=self.config['num_layers']
            ).to(self.device)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        X_tensor = torch.from_numpy(X_test).float().unsqueeze(0)
        pred_mean, pred_std = predict_with_uncertainty(
            self.model, X_tensor, self.device, num_passes=10
        )

        return pred_mean, pred_std

def main():
    """Entry point"""

    parser = argparse.ArgumentParser(description='Transformer Stock Predictor')
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip data fetching, use existing data from disk')
    args = parser.parse_args()

    config = load_config('config.yaml')
    pipeline = TrainingPipeline(config)

    # Train (fetches data automatically unless --skip-fetch)
    model = pipeline.train(epochs=int(config['epochs']), skip_fetch=args.skip_fetch)

    logger.info("Done!")

if __name__ == '__main__':
    main()

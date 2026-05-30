"""Create sequences from recent data with sentiment alignment for fine-tuning."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

from data.fetch import DataFetcher
from data.normalize import DataNormalizer
from sentiment.align_sentiment import SentimentAligner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sequences_with_sentiment(ohlcv_array, sentiment_array, seq_length=60):
    """Create sequences from OHLCV+sentiment data.

    Args:
        ohlcv_array (np.ndarray): Shape (N, 5) - OHLCV data
        sentiment_array (np.ndarray): Shape (N,) - sentiment values aligned to OHLCV
        seq_length (int): Length of each sequence

    Returns:
        tuple: (X, y_1d, y_5d, y_20d, y_vol) where X has shape (M, seq_length, 6)
    """
    if len(ohlcv_array) < seq_length + 20:
        return None, None, None, None, None

    X = []
    y_1d = []
    y_5d = []
    y_20d = []
    y_vol = []

    for i in range(len(ohlcv_array) - seq_length - 20):
        # Input: 60 timesteps of OHLCV + sentiment
        seq_ohlcv = ohlcv_array[i:i+seq_length, :]  # (60, 5)
        seq_sentiment = sentiment_array[i:i+seq_length]  # (60,)

        # Stack to (60, 6)
        seq_6d = np.column_stack([seq_ohlcv, seq_sentiment])
        X.append(seq_6d)

        # Targets: future close prices
        y_1d.append(ohlcv_array[i+seq_length, 3])      # 1 day ahead
        y_5d.append(ohlcv_array[i+seq_length+4, 3])    # 5 days ahead
        y_20d.append(ohlcv_array[i+seq_length+19, 3])  # 20 days ahead
        y_vol.append(sentiment_array[i+seq_length])    # TODO: should be volatility, using sentiment for now

    return (
        np.array(X, dtype=np.float32),
        np.array(y_1d, dtype=np.float32),
        np.array(y_5d, dtype=np.float32),
        np.array(y_20d, dtype=np.float32),
        np.array(y_vol, dtype=np.float32)
    )


def main():
    """Create sequences with sentiment for Feb-Mar 2026 fine-tuning."""

    print("="*70)
    print("PHASE 2: Create Sentiment-Enhanced Sequences")
    print("="*70)

    # Load configuration
    config = {'data_path': '../data'}
    fetcher = DataFetcher(data_path=config['data_path'])
    normalizer = DataNormalizer()
    aligner = SentimentAligner('data/sentiment/daily_sentiment.csv')

    # Load all tickers
    logger.info("Loading tickers from disk...")
    ticker_data = fetcher.load_tickers_grouped_from_disk()

    # Fit normalizer on all data
    logger.info("Fitting normalizer...")
    all_ohlcv = []
    for df in ticker_data.values():
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        available = [col for col in ohlcv_cols if col in df.columns]
        if len(available) == 5:
            ohlcv = df[available].values
            all_ohlcv.append(ohlcv)

    all_concat = np.concatenate(all_ohlcv)
    normalizer.fit_transform(all_concat)

    # Focus on Feb-Mar 2026 where sentiment data is available for higher quality
    seq_length = 60
    sentiment_min = pd.to_datetime('2026-02-20')  # When sentiment data starts
    sentiment_max = pd.to_datetime('2026-03-20')  # When sentiment data ends

    # Extend backward far enough to get 80+ trading days (seq_length + 20 buffer)
    # ~6 months back from Mar 20 = mid-September 2025
    min_date = sentiment_max - pd.Timedelta(days=180)  # ~180 calendar days ≈ 130+ trading days
    max_date = sentiment_max

    logger.info(f"Loading data from {min_date.date()} to {max_date.date()}")
    logger.info(f"Sentiment available {sentiment_min.date()} - {sentiment_max.date()}")
    logger.info(f"This focuses on high-quality sentiment signal period")

    # Filter to tickers with sentiment data for higher quality signal
    tickers_with_sentiment = set(aligner.sentiment_dict.keys())
    logger.info(f"Filtering to {len(tickers_with_sentiment)} tickers with sentiment data: {sorted(tickers_with_sentiment)}")

    train_X, train_y1d, train_y5d, train_y20d, train_yvol = [], [], [], [], []
    tickers_processed = 0
    tickers_skipped = 0
    sample_debug = 0

    for ticker, df in tqdm(ticker_data.items(), desc="Processing tickers", unit="ticker"):
        # Skip tickers without sentiment data
        if ticker not in tickers_with_sentiment:
            tickers_skipped += 1
            continue
        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns or 'Date' in df.columns:
                    date_col = 'date' if 'date' in df.columns else 'Date'
                    df.index = pd.to_datetime(df[date_col])
                else:
                    tickers_skipped += 1
                    continue

            # Filter to date range
            df_filtered = df[(df.index >= min_date) & (df.index <= max_date)]

            if sample_debug < 3:
                logger.info(f"{ticker}: total rows={len(df)}, filtered rows={len(df_filtered)}, min={df.index.min() if len(df) > 0 else 'N/A'}, index_type={type(df.index).__name__}")
                sample_debug += 1

            if len(df_filtered) < seq_length + 20:
                tickers_skipped += 1
                continue

            # Get OHLCV
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            available = [col for col in ohlcv_cols if col in df_filtered.columns]
            if len(available) < 5:
                tickers_skipped += 1
                continue

            ohlcv = df_filtered[available].values

            # Normalize
            ohlcv_normalized = normalizer.transform(ohlcv)

            # Get sentiment (aligned by date)
            sentiment = np.zeros(len(ohlcv_normalized))
            for j, date in enumerate(df_filtered.index):
                sentiment[j] = aligner.get_sentiment_for_date(ticker, date, default=0.0)

            # Create sequences
            X, y1d, y5d, y20d, yvol = create_sequences_with_sentiment(
                ohlcv_normalized, sentiment, seq_length
            )

            if X is not None and len(X) > 0:
                train_X.append(X)
                train_y1d.append(y1d)
                train_y5d.append(y5d)
                train_y20d.append(y20d)
                train_yvol.append(yvol)
                tickers_processed += 1

        except Exception as e:
            logger.debug(f"Error processing {ticker}: {e}")
            tickers_skipped += 1

    # Concatenate all
    if not train_X:
        logger.error(f"No sequences created! Processed {tickers_processed}, skipped {tickers_skipped}")
        return

    X_combined = np.concatenate(train_X, axis=0)
    y1d_combined = np.concatenate(train_y1d, axis=0)
    y5d_combined = np.concatenate(train_y5d, axis=0)
    y20d_combined = np.concatenate(train_y20d, axis=0)
    yvol_combined = np.concatenate(train_yvol, axis=0)

    logger.info(f"\nProcessed {tickers_processed} tickers, skipped {tickers_skipped}")
    logger.info(f"Created {X_combined.shape[0]} sequences")
    logger.info(f"Sequence shape: {X_combined.shape} (with 6 dimensions: OHLCV + sentiment)")

    # Save sequences
    output_dir = Path('data/sentiment/sequences_for_finetuning')
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'X.npy', X_combined)
    np.save(output_dir / 'y_1d.npy', y1d_combined)
    np.save(output_dir / 'y_5d.npy', y5d_combined)
    np.save(output_dir / 'y_20d.npy', y20d_combined)
    np.save(output_dir / 'y_vol.npy', yvol_combined)

    logger.info(f"\nSaved sequences to {output_dir}/")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY: Sentiment-Enhanced Sequences")
    print("="*70)
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Tickers processed: {tickers_processed}")
    print(f"Total sequences: {X_combined.shape[0]}")
    print(f"Sequence shape: {X_combined.shape}")
    print(f"\nFeatures (dimension 5 = sentiment):")
    print(f"  - Feature 0-4: OHLCV (from price data)")
    print(f"  - Feature 5: Sentiment (from Finnhub + Reddit sources)")
    print(f"\nSentiment statistics in sequences:")
    print(f"  Mean: {X_combined[:, :, 5].mean():.4f}")
    print(f"  Std: {X_combined[:, :, 5].std():.4f}")
    print(f"  Min: {X_combined[:, :, 5].min():.4f}")
    print(f"  Max: {X_combined[:, :, 5].max():.4f}")

    return X_combined, y1d_combined, y5d_combined, y20d_combined, yvol_combined


if __name__ == '__main__':
    main()

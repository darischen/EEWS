#!/usr/bin/env python3
"""
Simple CLI for single-ticker stock price forecasting.

Usage:
    python src/cli.py AAPL
    python src/cli.py MSFT --show-chart
    python src/cli.py SPY --days 20
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from .models.transformer import DeepStockTransformer
from .data.normalize import DataNormalizer
from .inference.predict import predict_single_ticker
from .plot import plot_forecast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path='checkpoints/best_transformer.pth', device='cpu'):
    """Load trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            f"Please run: python src/main.py --epochs 50"
        )

    model = DeepStockTransformer(
        input_dim=5,
        seq_length=60,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.15
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info(f"✓ Loaded model from {checkpoint_path}")
    return model


def load_ticker_data(ticker, data_path='../data'):
    """Load historical data for a single ticker"""
    # Check stocks folder
    stocks_path = os.path.join(data_path, 'stocks', f'{ticker}.csv')
    etfs_path = os.path.join(data_path, 'etfs', f'{ticker}.csv')

    if os.path.exists(stocks_path):
        csv_path = stocks_path
    elif os.path.exists(etfs_path):
        csv_path = etfs_path
    else:
        raise FileNotFoundError(
            f"Ticker '{ticker}' not found in {data_path}/stocks or /etfs\n"
            f"Run src/main.py first to fetch and update data."
        )

    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    df.columns = df.columns.str.lower()

    # Keep only OHLCV
    required = ['open', 'high', 'low', 'close', 'volume']
    df = df[[col for col in required if col in df.columns]]
    df = df.dropna()
    df = df.sort_index()

    logger.info(f"✓ Loaded {len(df)} rows of {ticker} data")
    logger.info(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    return df


def create_sequence(data, seq_length=60):
    """Create input sequence from normalized data"""
    if len(data) < seq_length:
        raise ValueError(
            f"Not enough data for {ticker}. "
            f"Need at least {seq_length} days, got {len(data)}"
        )

    # Use last seq_length rows
    sequence = data[-seq_length:].values
    return np.array([sequence], dtype=np.float32)  # Add batch dimension


def main():
    parser = argparse.ArgumentParser(
        description='Stock price forecast using Transformer model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/cli.py AAPL              # Show forecast for Apple
  python src/cli.py MSFT --chart      # Include charts
  python src/cli.py SPY --mc 20       # Use 20 MC dropout passes for uncertainty
  python src/cli.py GOOGL --save pred.png  # Save chart to file
        """
    )

    parser.add_argument(
        'ticker',
        help='Stock ticker symbol (e.g., AAPL, MSFT, SPY)'
    )
    parser.add_argument(
        '--chart',
        action='store_true',
        help='Display matplotlib chart'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save chart to file (e.g., forecast.png)'
    )
    parser.add_argument(
        '--mc',
        type=int,
        default=10,
        help='Number of MC dropout passes for uncertainty (default: 10)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Compute device: cpu or cuda (default: cpu)'
    )

    args = parser.parse_args()

    try:
        ticker = args.ticker.upper()
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        logger.info(f"\n{'='*60}")
        logger.info(f"Stock Price Forecast: {ticker}")
        logger.info(f"{'='*60}\n")

        # Load model
        model = load_model(device=device)

        # Load ticker data
        df = load_ticker_data(ticker)

        # Normalize data
        normalizer = DataNormalizer()
        normalized = normalizer.fit_transform(df.values)

        logger.info(f"✓ Normalized data (mean=0, std=1)")

        # Create sequence
        X = create_sequence(normalized)
        X_tensor = torch.from_numpy(X).float().to(device)

        logger.info(f"\n{'─'*60}")
        logger.info(f"Generating Forecast (MC passes: {args.mc})")
        logger.info(f"{'─'*60}\n")

        # Make predictions
        predictions, uncertainties = predict_single_ticker(
            model, X_tensor, device, num_passes=args.mc
        )

        # Denormalize predictions (only close price column is index 3)
        # We need to denormalize just the 4th column (close price)
        close_idx = 3
        close_scaler_mean = normalizer.scaler.mean_[close_idx]
        close_scaler_std = normalizer.scaler.scale_[close_idx]

        def denorm_price(norm_val):
            return norm_val * close_scaler_std + close_scaler_mean

        # Extract predictions
        pred_1day = denorm_price(predictions['1day'].item())
        pred_5day = denorm_price(predictions['5day'].item())
        pred_20day = denorm_price(predictions['20day'].item())

        unc_1day = uncertainties['1day'].item() * close_scaler_std
        unc_5day = uncertainties['5day'].item() * close_scaler_std
        unc_20day = uncertainties['20day'].item() * close_scaler_std

        vol = predictions['volatility'].item()

        # Current price
        current_price = df['close'].iloc[-1]
        current_date = df.index[-1].date()

        # Calculate changes
        chg_1day = ((pred_1day - current_price) / current_price) * 100
        chg_5day = ((pred_5day - current_price) / current_price) * 100
        chg_20day = ((pred_20day - current_price) / current_price) * 100

        # Display results
        logger.info(f"Current Price ({current_date}): ${current_price:.2f}")
        logger.info(f"\n{'Horizon':<12} {'Forecast':<15} {'Change':<12} {'Uncertainty':<15}")
        logger.info(f"{'-'*60}")
        logger.info(
            f"{'1-Day':<12} ${pred_1day:>8.2f}      "
            f"{chg_1day:>+7.2f}%    "
            f"± ${unc_1day:.2f}"
        )
        logger.info(
            f"{'5-Day':<12} ${pred_5day:>8.2f}      "
            f"{chg_5day:>+7.2f}%    "
            f"± ${unc_5day:.2f}"
        )
        logger.info(
            f"{'20-Day':<12} ${pred_20day:>8.2f}      "
            f"{chg_20day:>+7.2f}%    "
            f"± ${unc_20day:.2f}"
        )
        logger.info(f"\nVolatility: {vol:.4f} ({vol*100:.2f}%)")

        # Plot if requested
        if args.chart or args.save:
            logger.info(f"\n{'─'*60}")
            logger.info("Generating charts...")
            logger.info(f"{'─'*60}\n")

            plot_forecast(
                ticker=ticker,
                df=df,
                current_price=current_price,
                pred_1day=pred_1day,
                pred_5day=pred_5day,
                pred_20day=pred_20day,
                unc_1day=unc_1day,
                unc_5day=unc_5day,
                unc_20day=unc_20day,
                show=args.chart,
                save_path=args.save
            )

            if args.save:
                logger.info(f"✓ Chart saved to {args.save}")

        logger.info(f"\n{'='*60}\n")

    except FileNotFoundError as e:
        logger.error(f"\n❌ Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

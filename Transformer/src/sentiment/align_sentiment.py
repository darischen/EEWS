"""Align sentiment data with price sequences."""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAligner:
    """Align sentiment data with trading sequences."""

    def __init__(self, sentiment_csv='data/sentiment/daily_sentiment.csv'):
        """Load sentiment data.

        Args:
            sentiment_csv (str): Path to daily_sentiment.csv (from Finnhub and Reddit sources)
        """
        self.sentiment_df = pd.read_csv(sentiment_csv)
        self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])

        # Create lookup dict: {ticker: {date: sentiment}}
        self.sentiment_dict = {}
        for ticker in self.sentiment_df['ticker'].unique():
            ticker_data = self.sentiment_df[self.sentiment_df['ticker'] == ticker]
            self.sentiment_dict[ticker] = dict(
                zip(ticker_data['date'].dt.date, ticker_data['sentiment'])
            )

        logger.info(f"Loaded sentiment for {len(self.sentiment_dict)} tickers")
        logger.info(f"Date range: {self.sentiment_df['date'].min()} to {self.sentiment_df['date'].max()}")

    def get_sentiment_for_date(self, ticker, date, default=0.0):
        """Get sentiment for a ticker on a given date.

        Args:
            ticker (str): Stock ticker
            date (datetime or str): Trading date
            default (float): Default value if no sentiment found (neutral=0.0)

        Returns:
            float: Sentiment value in [-1, 1]
        """
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        elif hasattr(date, 'date'):
            date = date.date()
        else:
            date = pd.Timestamp(date).date()

        if ticker not in self.sentiment_dict:
            return default

        return self.sentiment_dict[ticker].get(date, default)

    def extend_sequences_with_sentiment(self, X, dates_per_ticker, ticker_list):
        """Extend price sequences with sentiment as 6th feature.

        Args:
            X (np.ndarray): Shape (N, 60, 5) - OHLCV sequences
            dates_per_ticker (dict): {ticker: [dates_per_sequence]}
            ticker_list (list): Ticker for each sequence

        Returns:
            np.ndarray: Shape (N, 60, 6) - OHLCV + sentiment
        """
        n_sequences = X.shape[0]
        X_extended = np.zeros((n_sequences, X.shape[1], 6), dtype=np.float32)

        # Copy OHLCV
        X_extended[:, :, :5] = X

        # Add sentiment as 6th feature
        for i in range(n_sequences):
            ticker = ticker_list[i]
            dates = dates_per_ticker[ticker][i] if i < len(dates_per_ticker[ticker]) else None

            if dates is not None:
                # Use the last date in the sequence as sentiment reference
                # (the date we're trying to predict for)
                last_date = dates[-1]
                sentiment = self.get_sentiment_for_date(ticker, last_date, default=0.0)
            else:
                sentiment = 0.0

            # Broadcast sentiment across all 60 timesteps in the sequence
            X_extended[i, :, 5] = sentiment

        logger.info(f"Extended {n_sequences} sequences from 5→6 dimensions with sentiment")
        return X_extended


def forward_fill_missing_sentiment(df, date_range):
    """Forward-fill missing dates in sentiment data.

    Args:
        df (pd.DataFrame): Sentiment data with columns [ticker, date, sentiment]
        date_range (tuple): (start_date, end_date) as strings 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: Sentiment data with all trading dates, forward-filled
    """
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    # Create all trading dates (business days only)
    all_dates = pd.bdate_range(start=start, end=end)

    # For each ticker, ensure all trading dates exist
    filled_rows = []

    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])

        # Reindex to all trading dates and forward-fill
        ticker_filled = ticker_data.set_index('date').reindex(
            all_dates, method='ffill'
        ).reset_index()

        ticker_filled['ticker'] = ticker
        filled_rows.append(ticker_filled)

    result = pd.concat(filled_rows, ignore_index=True)
    logger.info(f"Forward-filled sentiment from {len(df)} to {len(result)} rows")

    return result


def normalize_sentiment(sentiment_values, method='minmax'):
    """Normalize sentiment to consistent scale.

    Args:
        sentiment_values (np.ndarray): Raw sentiment values
        method (str): 'minmax' ([-1,1]) or 'zscore' (0-mean, 1-std)

    Returns:
        np.ndarray: Normalized sentiment
    """
    if method == 'minmax':
        # Already in [-1, 1] from TextBlob, just ensure bounds
        return np.clip(sentiment_values, -1.0, 1.0)

    elif method == 'zscore':
        # Standardize to mean=0, std=1 like OHLCV
        mean = np.nanmean(sentiment_values)
        std = np.nanstd(sentiment_values)
        if std == 0:
            std = 1.0
        return (sentiment_values - mean) / std

    return sentiment_values


if __name__ == '__main__':
    # Example usage
    logger.info("SentimentAligner module ready for use")
    logger.info("Import and use: from sentiment.align_sentiment import SentimentAligner")

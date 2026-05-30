import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataFetcher:
    """Fetch OHLCV, VIX, and sentiment. Auto-discovers all tickers from CSV files and updates them."""

    def __init__(self, data_path='../data'):
        self.data_path = data_path

    def discover_all_tickers(self):
        """Auto-discover all tickers from existing CSV files in ../data/stocks and ../data/etfs"""
        tickers = []

        # Check stocks folder
        stocks_dir = os.path.join(self.data_path, 'stocks')
        if os.path.exists(stocks_dir):
            stock_files = [f for f in os.listdir(stocks_dir) if f.endswith('.csv')]
            for file in tqdm(stock_files, desc="Scanning stocks/", unit="file"):
                ticker = file.replace('.csv', '').upper()
                tickers.append(ticker)

        # Check etfs folder
        etfs_dir = os.path.join(self.data_path, 'etfs')
        if os.path.exists(etfs_dir):
            etf_files = [f for f in os.listdir(etfs_dir) if f.endswith('.csv')]
            for file in tqdm(etf_files, desc="Scanning etfs/", unit="file"):
                ticker = file.replace('.csv', '').upper()
                tickers.append(ticker)

        if not tickers:
            raise ValueError(f"No CSV files found in {self.data_path}/stocks or /etfs")

        tickers = sorted(list(set(tickers)))  # Remove duplicates, sort
        logger.info(f"Discovered {len(tickers)} total tickers")
        return tickers

    def get_csv_path(self, ticker):
        """Get path to ticker's CSV file (stocks or etfs folder)"""
        stocks_path = os.path.join(self.data_path, 'stocks', f'{ticker}.csv')
        etfs_path = os.path.join(self.data_path, 'etfs', f'{ticker}.csv')

        # Check which folder it's in
        if os.path.exists(stocks_path):
            return stocks_path
        elif os.path.exists(etfs_path):
            return etfs_path
        else:
            # If not found, assume stocks (for new tickers)
            return stocks_path

    def get_last_date_from_csv(self, ticker):
        """Get the most recent date in existing CSV file"""
        csv_path = self.get_csv_path(ticker)

        if not os.path.exists(csv_path):
            return None

        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                return None

            # Try different column names for date
            date_col = None
            for col in ['Date', 'date', 'DATE']:
                if col in df.columns:
                    date_col = col
                    break

            if date_col is None:
                logger.warning(f"No date column found in {csv_path}, will refetch all")
                return None

            last_date = pd.to_datetime(df[date_col].iloc[-1])
            logger.info(f"{ticker}: Last date in CSV: {last_date.date()}")
            return last_date

        except Exception as e:
            logger.warning(f"Error reading {csv_path}: {e}, will refetch all")
            return None

    def fetch_stock_and_update(self, ticker):
        """Fetch ticker from Yahoo and update existing CSV file"""
        try:
            csv_path = self.get_csv_path(ticker)
            last_date = self.get_last_date_from_csv(ticker)

            # Determine fetch start date
            if last_date:
                # Only fetch from day after last date
                start_date = last_date + timedelta(days=1)
            else:
                # Fetch last 10 years if file doesn't exist
                start_date = datetime.now() - timedelta(days=365*10)

            end_date = datetime.now()

            # Skip if CSV is already up to date
            if start_date > end_date:
                logger.info(f"{ticker}: Already up to date (last date: {last_date.date()})")
                return True

            logger.info(f"Fetching {ticker} from {start_date.date()} to {end_date.date()}")

            # Fetch from Yahoo
            df_new = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df_new.empty:
                logger.warning(f"{ticker}: No new data from Yahoo")
                return False

            # Handle MultiIndex columns (flatten if present - newer yfinance may return this)
            if isinstance(df_new.columns, pd.MultiIndex):
                # Get column names at the deepest level
                df_new.columns = df_new.columns.get_level_values(0)

            # Standardize column names (handle case variations)
            df_new.columns = df_new.columns.str.lower().str.strip()

            # Keep only OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df_new.columns]

            if len(available_cols) < 5:
                logger.warning(f"{ticker}: Missing required columns (found: {list(df_new.columns)}), skipping")
                return False

            df_new = df_new[available_cols].copy()

            logger.info(f"{ticker}: Fetched {len(df_new)} new rows")

            # Load existing data if file exists
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
                df_existing.columns = df_existing.columns.str.lower()

                # Combine (remove duplicates, keep newer)
                df_combined = pd.concat([df_existing, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined = df_combined.sort_index()

                logger.info(f"{ticker}: Combined to {len(df_combined)} total rows")
            else:
                df_combined = df_new.sort_index()
                logger.info(f"{ticker}: Created new file with {len(df_combined)} rows")

            # Save back to CSV
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df_combined.to_csv(csv_path)
            logger.info(f"{ticker}: Saved to {csv_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to fetch/update {ticker}: {e}")
            return False

    def fetch_and_update_all_tickers(self):
        """Auto-discover all tickers and update them with latest Yahoo data"""
        tickers = self.discover_all_tickers()

        logger.info(f"\nUpdating {len(tickers)} tickers...")
        logger.info("=" * 80)

        success_count = 0
        failed_tickers = []

        for ticker in tqdm(tickers, desc="Updating tickers", unit="ticker"):
            if self.fetch_stock_and_update(ticker):
                success_count += 1
            else:
                failed_tickers.append(ticker)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Update complete: {success_count}/{len(tickers)} succeeded")

        if failed_tickers:
            logger.warning(f"Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers[:10])}")
            if len(failed_tickers) > 10:
                logger.warning(f"... and {len(failed_tickers) - 10} more")

        return tickers, success_count, failed_tickers

    def fetch_vix(self, years=10):
        """Fetch VIX volatility index"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)

        try:
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            vix = vix[['Close']].copy()
            vix.columns = ['vix']

            logger.info(f"Fetched VIX: {len(vix)} rows")
            return vix

        except Exception as e:
            logger.error(f"Failed to fetch VIX: {e}")
            return None

def load_tickers_from_disk(self, tickers=None):
        """Load already-fetched ticker data from disk (after update)"""
        if tickers is None:
            tickers = self.discover_all_tickers()

        all_data = []

        for ticker in tqdm(tickers, desc="Loading ticker CSVs", unit="ticker"):
            csv_path = self.get_csv_path(ticker)

            if not os.path.exists(csv_path):
                logger.warning(f"CSV not found for {ticker}")
                continue

            try:
                df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
                df.columns = df.columns.str.lower()
                df['ticker'] = ticker
                all_data.append(df)

            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")

        if not all_data:
            raise ValueError("No ticker data loaded")

        combined = pd.concat(all_data).sort_index()
        logger.info(f"Combined {len(all_data)} tickers: {len(combined)} total rows")

        return combined

    def load_tickers_grouped_from_disk(self, tickers=None, min_last_trade_days=365):
        """Load ticker data from disk as a dict keyed by ticker symbol.

        Returns dict[str, DataFrame] instead of one concatenated DataFrame,
        so sequences can be created per-ticker without cross-ticker contamination.

        Args:
            tickers: List of tickers to load. If None, discovers all.
            min_last_trade_days: Skip tickers not traded in this many days (filters delisted).
                Default 365 (1 year). Set to None to disable freshness check.
        """
        if tickers is None:
            tickers = self.discover_all_tickers()

        ticker_data = {}
        stale_count = 0
        min_date = datetime.now() - timedelta(days=min_last_trade_days) if min_last_trade_days else None

        for ticker in tqdm(tickers, desc="Loading ticker CSVs (grouped)", unit="ticker"):
            csv_path = self.get_csv_path(ticker)

            if not os.path.exists(csv_path):
                logger.warning(f"CSV not found for {ticker}")
                continue

            try:
                df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
                df.columns = df.columns.str.lower()

                # Filter out stale/delisted tickers
                if min_date is not None and len(df) > 0:
                    last_trade_date = df.index[-1]
                    if last_trade_date < min_date:
                        logger.debug(f"{ticker}: Stale (last trade: {last_trade_date.date()}), skipping")
                        stale_count += 1
                        continue

                ticker_data[ticker] = df

            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")

        if not ticker_data:
            raise ValueError("No ticker data loaded")

        logger.info(f"Loaded {len(ticker_data)} tickers (grouped), skipped {stale_count} stale")
        return ticker_data

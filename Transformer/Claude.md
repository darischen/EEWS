# Transformer Stock Predictor: Manual Training + Auto Data Updates

**Objective**: Transformer-based stock prediction with sentiment scraping. Manual training/retraining with automatic data fetching before each run.

**Structure**: Standalone `transformers/` folder. Reuses EEWS data. No cloud setup, no auto daemon.

---

## 1. Architecture Overview

### Data Flow
```
Yahoo Finance (daily OHLCV)  →  NewsAPI (daily sentiment)  →  CSV/Database
                                                                    ↓
                                                        (Auto-fetch on train)
                                                                    ↓
                                                        Normalize + Sequences
                                                                    ↓
                                                        DeepStockTransformer
                                                                    ↓
                                                        Save predictions + model
```

### Manual Workflow
```
User: python train.py
    ↓
Fetch latest data from Yahoo + NewsAPI (automatic)
    ↓
Load EEWS data (reuse)
    ↓
Combine + normalize
    ↓
Create sequences
    ↓
Train transformer (you decide epochs)
    ↓
Save checkpoint + predictions
    ↓
User: Done, next manual run whenever
```

**No daemon, no scheduler, no cloud. Just:**
```bash
python src/main.py  # Fetches latest data, trains, saves
```

**Progress tracking**: All long-running operations (data loading, fetching, training, etc.) use `tqdm` progress bars for real-time feedback.

---

## 2. Repository Structure

```
EEWS/
├── data/                              # Shared data (5,884 stocks + 2,164 ETFs)
│   ├── stocks/
│   │   ├── A.csv, AA.csv, ..., AAPL.csv, ..., ZZZ.csv
│   │   └── (5,884 individual stock files)
│   └── etfs/
│       ├── AAAU.csv, AADR.csv, ..., ZZZ.csv
│       └── (2,164 individual ETF files)
│
├── Transformer/                       # NEW: Fresh transformer implementation
│   ├── README.md                      # Setup instructions
│   ├── requirements.txt               # Dependencies
│   ├── .env.example                   # API keys template
│   ├── config.yaml                    # Hyperparameters
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── main.py                    # Entry point
│   │   │
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── fetch.py               # Yahoo Finance + NewsAPI
│   │   │   ├── normalize.py           # StandardScaler + sequences
│   │   │   └── loader.py              # Load from ../data/stocks and ../data/etfs
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── transformer.py         # DeepStockTransformer
│   │   │   └── loss.py                # Multi-task loss
│   │   │
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── train.py               # Training loop
│   │   │   ├── warmup.py              # LR scheduling
│   │   │   └── checkpoint.py          # Save/load
│   │   │
│   │   └── inference/
│   │       ├── __init__.py
│   │       ├── predict.py             # Batch prediction
│   │       └── uncertainty.py         # MC dropout
│   │
│   ├── checkpoints/
│   │   └── .gitkeep
│   │
│   ├── logs/
│   │   └── .gitkeep
│   │
│   ├── data/
│   │   ├── raw/                       # Downloaded from Yahoo/NewsAPI
│   │   │   └── .gitkeep
│   │   └── processed/                 # Normalized data
│   │       └── .gitkeep
│   │
│   └── notebooks/
│       ├── 01_eda.ipynb               # Exploratory analysis
│       └── 02_backtest.ipynb          # Validation
│
└── LSTM/                              # OLD: Legacy code (untouched)
    └── (previous implementations)
```

---

## 3. Data Fetching with Automatic Updates

---

## 3.1b Auto-Discovery & Update Workflow

### How It Works

When you run `python src/main.py`:

1. **Auto-discover all tickers**
   ```python
   tickers = fetcher.discover_all_tickers()
   ```
   - Scans `../data/stocks/` for all `*.csv` files
   - Scans `../data/etfs/` for all `*.csv` files
   - Extracts ticker names (filename without .csv)
   - Returns sorted list of ALL tickers

2. **Update each ticker with latest data**
   ```python
   fetcher.fetch_and_update_all_tickers()
   ```
   For EACH ticker:
   - Reads CSV file to get last date
   - Fetches from Yahoo from (last_date + 1 day) to today
   - Combines new data with existing data
   - Removes duplicates, keeps newest
   - Saves back to same CSV file

3. **Load all updated data from disk**
   ```python
   combined = fetcher.load_tickers_from_disk()
   ```
   - Loads all updated CSV files from disk
   - Combines into single DataFrame
   - Ready for training

### Example

Your eews/data structure:
```
../data/
├── stocks/
│   ├── AAPL.csv       (880 rows, last date: 2024-01-15)
│   ├── MSFT.csv       (875 rows, last date: 2024-01-14)
│   ├── GOOGL.csv      (882 rows, last date: 2024-01-16)
│   └── ... (hundreds more)
└── etfs/
    ├── SPY.csv        (800 rows, last date: 2024-01-15)
    ├── QQQ.csv        (798 rows, last date: 2024-01-14)
    └── ... (hundreds more)
```

When you run `python src/main.py`:

```
Discovering tickers...
✓ Found AAPL
✓ Found MSFT
✓ Found GOOGL
... (scanning all files)
✓ Found SPY (ETF)
✓ Found QQQ (ETF)
Discovered 1,847 tickers

Updating tickers:
[1/1847] AAPL: Last date in CSV: 2024-01-15
         Fetching from 2024-01-16 to 2024-03-09
         Fetched 40 new rows
         Combined to 920 rows total
         Saved to ../data/stocks/AAPL.csv ✓

[2/1847] MSFT: Last date in CSV: 2024-01-14
         Fetching from 2024-01-15 to 2024-03-09
         Fetched 41 new rows
         Combined to 916 rows total
         Saved to ../data/stocks/MSFT.csv ✓

[3/1847] GOOGL: Last date in CSV: 2024-01-16
         Fetching from 2024-01-17 to 2024-03-09
         Fetched 39 new rows
         Combined to 921 rows total
         Saved to ../data/stocks/GOOGL.csv ✓

... (continuing for all 1,847 tickers)

Update complete: 1847/1847 succeeded
All tickers now have data through 2024-03-09
```

### Important: Incremental Updates (After First Run)

After the first run, **each subsequent run only fetches new data since the last date in the CSV**.

```
Run 1 (2024-03-09): 
  AAPL.csv before: last date 2024-01-15
  Fetches: 2024-01-16 to 2024-03-09 (40 rows)
  AAPL.csv after: last date 2024-03-09
  
Run 2 (2024-03-10):
  AAPL.csv before: last date 2024-03-09
  Fetches: 2024-03-10 to 2024-03-10 (1 row only!)
  AAPL.csv after: last date 2024-03-10
  
Run 3 (2024-03-11):
  AAPL.csv before: last date 2024-03-10
  Fetches: 2024-03-11 to 2024-03-11 (1 row only!)
  AAPL.csv after: last date 2024-03-11
```

The key code that makes this efficient:

```python
def fetch_stock_and_update(self, ticker):
    last_date = get_last_date_from_csv(ticker)  # Reads CSV
    
    # ONLY fetch from day after last date
    start_date = last_date + timedelta(days=1)
    end_date = datetime.now()
    
    df_new = yf.download(ticker, start=start_date, end=end_date)
    # Yahoo only returns new rows
    
    # Combine with existing data
    df_combined = pd.concat([df_existing, df_new])
    
    # Save back to CSV (now has latest data)
    df_combined.to_csv(csv_path)
```

**Result**: Each run auto-discovers all tickers, updates only the new data since last run, and saves back to CSV. No wasted API calls.

Loading all updated data from disk...
Loaded 1847 tickers, 3.2M total rows

Ready for training!
```

### Result After Update

Each CSV file contains:
- Original old data (from your previous download)
- New data from Yahoo (since last date)
- Combined and deduplicated
- Saved back to same file

Next time you run `python src/main.py`, it only fetches NEW data (since the last run).

`src/data/fetch.py`:

```python
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from textblob import TextBlob
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataFetcher:
    """Fetch OHLCV, VIX, and sentiment. Auto-discovers all tickers from CSV files and updates them."""
    
    def __init__(self, data_path='../data', cache_dir='./data/raw'):
        self.data_path = data_path
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
    
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
        stocks_path = os.path.join(self.eews_data_path, 'stocks', f'{ticker}.csv')
        etfs_path = os.path.join(self.eews_data_path, 'etfs', f'{ticker}.csv')
        
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
            df = pd.read_csv(csv_path, nrows=1)  # Read just header + 1 row
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
            
            logger.info(f"Fetching {ticker} from {start_date.date()} to {end_date.date()}")
            
            # Fetch from Yahoo
            df_new = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df_new.empty:
                logger.warning(f"{ticker}: No new data from Yahoo")
                return False
            
            # Standardize column names
            df_new = df_new[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df_new.columns = df_new.columns.str.lower()
            
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
    
    def fetch_sentiment(self, ticker, days=1):
        """Fetch sentiment from NewsAPI"""
        if not self.newsapi_key:
            logger.warning("NEWSAPI_KEY not set. Using neutral sentiment.")
            return 0.5
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            params = {
                'q': f'"{ticker}" stock',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': self.newsapi_key,
                'pageSize': 100,
                'language': 'en'
            }
            
            response = requests.get('https://newsapi.org/v2/everything', 
                                   params=params, timeout=10)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            
            if not articles:
                logger.warning(f"No articles for {ticker}, returning neutral")
                return 0.5
            
            scores = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                blob = TextBlob(text)
                polarity = (blob.sentiment.polarity + 1) / 2  # Convert -1,1 to 0,1
                scores.append(polarity)
            
            avg_sentiment = sum(scores) / len(scores)
            logger.info(f"{ticker} sentiment: {avg_sentiment:.3f}")
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"Sentiment fetch failed for {ticker}: {e}")
            return 0.5
    
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
```

### 3.2 Load and Combine EEWS Data

`src/data/loader.py`:

```python
import os
import pandas as pd
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EEWSDataLoader:
    """Load existing EEWS data from eews/ folder"""
    
    def __init__(self, data_path='../data/stocks'):
        self.data_path = data_path
    
    def load_single_file(self, file_path):
        """Load single CSV, standardize columns"""
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Ensure date column
        if 'date' not in df.columns:
            raise ValueError(f"No date column in {file_path}")
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        # Keep only OHLCV
        required = ['open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required if col in df.columns]]
        
        return df
    
    def load_all_eews_data(self):
        """Load all CSV files from data/ directory"""
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        all_data = []

        for file in tqdm(csv_files, desc="Loading data CSVs", unit="file"):
            try:
                path = os.path.join(self.data_path, file)
                df = self.load_single_file(path)
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Skipped {file}: {e}")

        if not all_data:
            raise ValueError("No data loaded")

        combined = pd.concat(all_data).sort_index()
        logger.info(f"Combined {len(all_data)} files: {len(combined)} total rows")
        return combined
```

### 3.3 Normalization + Sequences

`src/data/normalize.py`:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm

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

def create_sequences(data, seq_length=60):
    """Create sliding window sequences with multi-horizon targets"""
    X, y_1d, y_5d, y_20d, y_vol = [], [], [], [], []

    n_sequences = len(data) - seq_length - 20
    for i in tqdm(range(n_sequences), desc="Creating sequences", unit="seq"):
        X.append(data[i:i+seq_length])

        # Multi-horizon targets
        y_1d.append(data[i+seq_length, 3])      # Close price next day
        y_5d.append(data[i+seq_length+4, 3])    # Close 5 days out
        y_20d.append(data[i+seq_length+19, 3])  # Close 20 days out

        # Volatility (std of close prices in next 5 days)
        y_vol.append(np.std(data[i+seq_length:i+seq_length+5, 3]))

    return (np.array(X, dtype=np.float32),
            np.array(y_1d, dtype=np.float32),
            np.array(y_5d, dtype=np.float32),
            np.array(y_20d, dtype=np.float32),
            np.array(y_vol, dtype=np.float32))
```

---

## 4. DeepStockTransformer Model

`src/models/transformer.py`:

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Learnable absolute positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """x shape: (batch, seq_length, d_model)"""
        return x + self.pe[:, :x.size(1), :]

class DeepStockTransformer(nn.Module):
    """Transformer for multi-horizon stock prediction"""
    
    def __init__(self, input_dim=6, seq_length=60, d_model=128, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.15):
        super(DeepStockTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length)
        
        # Transformer encoder (4 layers, 8 heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Multi-horizon prediction heads
        self.head_1day = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.head_5day = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.head_20day = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Volatility auxiliary task
        self.head_volatility = nn.Linear(d_model, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_length, input_dim)
        
        Returns:
            dict with predictions for each horizon
        """
        # Project to embedding dimension
        x = self.input_projection(x)  # (batch, seq_length, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_length, d_model)
        
        # Take last token
        x_last = x[:, -1, :]  # (batch, d_model)
        
        # Multi-horizon predictions
        return {
            '1day': self.head_1day(x_last),
            '5day': self.head_5day(x_last),
            '20day': self.head_20day(x_last),
            'volatility': self.head_volatility(x_last)
        }
```

### Loss Function

`src/models/loss.py`:

```python
import torch
import torch.nn as nn

def multi_task_loss(predictions, targets, weights=None):
    """Multi-task loss: 1day + 5day + 20day + volatility"""
    
    if weights is None:
        weights = {
            '1day': 0.5,
            '5day': 0.25,
            '20day': 0.15,
            'volatility': 0.1
        }
    
    mse = nn.MSELoss()
    
    loss_1day = mse(predictions['1day'].squeeze(), targets['1day'])
    loss_5day = mse(predictions['5day'].squeeze(), targets['5day'])
    loss_20day = mse(predictions['20day'].squeeze(), targets['20day'])
    loss_vol = mse(predictions['volatility'].squeeze(), targets['volatility'])
    
    total_loss = (weights['1day'] * loss_1day + 
                  weights['5day'] * loss_5day + 
                  weights['20day'] * loss_20day + 
                  weights['volatility'] * loss_vol)
    
    return total_loss, {
        '1day': loss_1day.item(),
        '5day': loss_5day.item(),
        '20day': loss_20day.item(),
        'volatility': loss_vol.item()
    }
```

---

## 5. Training Loop (Manual)

`src/training/train.py`:

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def get_lr_scheduler(optimizer, epochs, warmup_epochs=5):
    """Cosine annealing with linear warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, train_loader, optimizer, loss_fn, device, scaler):
    """Train single epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_x, targets in pbar:
        batch_x = batch_x.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        optimizer.zero_grad()
        
        with autocast():
            predictions = model(batch_x)
            loss, _ = loss_fn(predictions, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, loss_fn, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for batch_x, targets in pbar:
            batch_x = batch_x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            with autocast():
                predictions = model(batch_x)
                loss, _ = loss_fn(predictions, targets)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, device, epochs=50, lr=5e-4, 
                loss_fn=None, checkpoint_path='best_model.pth', patience=15):
    """Full training loop with early stopping"""
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = get_lr_scheduler(optimizer, epochs, warmup_epochs=5)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting training: {epochs} epochs, lr={lr}")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"✓ New best model saved")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.6f}")
    
    return model, {'train': train_losses, 'val': val_losses}
```

---

## 6. Inference with Uncertainty

`src/inference/predict.py`:

```python
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def predict_batch(model, X, device):
    """Single prediction"""
    model.eval()
    with torch.no_grad():
        output = model(X.to(device))
    
    return {k: v.cpu().numpy() for k, v in output.items()}

def predict_with_uncertainty(model, X, device, num_passes=10):
    """Monte Carlo dropout for uncertainty estimation"""
    
    model.train()  # Keep dropout enabled
    predictions_1day = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            output = model(X.to(device))
            predictions_1day.append(output['1day'].cpu().numpy())
    
    predictions_1day = np.concatenate(predictions_1day, axis=0)
    mean_pred = np.mean(predictions_1day)
    std_pred = np.std(predictions_1day)
    
    logger.info(f"Prediction: {mean_pred:.6f} ± {std_pred:.6f}")
    
    return mean_pred, std_pred
```

---

## 7. Main Training Script

`src/main.py`:

```python
import os
import torch
import yaml
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

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
from src.data.fetch import DataFetcher
from src.data.loader import EEWSDataLoader
from src.data.normalize import DataNormalizer, create_sequences
from src.models.transformer import DeepStockTransformer
from src.models.loss import multi_task_loss
from src.training.train import train_model
from src.inference.predict import predict_with_uncertainty
from src.training.checkpoint import load_checkpoint, save_checkpoint

import torch.utils.data as data_utils

def load_config(path='config.yaml'):
    """Load configuration"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

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
        """Fetch and update ALL tickers from Yahoo Finance"""
        logger.info("Fetching and updating ALL tickers from Yahoo Finance...")
        
        # Auto-discover all tickers from ../data/stocks and ../data/etfs
        # This also updates the CSV files with latest data
        tickers, success_count, failed = self.fetcher.fetch_and_update_all_tickers()
        
        logger.info(f"Successfully updated {success_count}/{len(tickers)} tickers")
        
        # Now load all the updated ticker data from disk
        logger.info("Loading all updated ticker data...")
        combined = self.fetcher.load_tickers_from_disk(tickers)
        
        logger.info(f"Combined data shape: {combined.shape}")
        logger.info(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
        
        return combined
    
    def prepare_data(self, data):
        """Normalize and create sequences"""
        logger.info("Preparing data...")

        # Remove NaN
        logger.info("Removing NaN values...")
        data = data.fillna(method='ffill').dropna()
        logger.info(f"Data shape after NaN removal: {data.shape}")

        # Extract OHLCV (ignore ticker column if present)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        ohlcv = data[[col for col in ohlcv_cols if col in data.columns]]

        # Normalize
        logger.info("Normalizing features...")
        normalized = self.normalizer.fit_transform(ohlcv.values)

        # Create sequences
        logger.info(f"Creating sequences (length={self.config['seq_length']})...")
        X, y_1d, y_5d, y_20d, y_vol = create_sequences(
            normalized,
            seq_length=self.config['seq_length']
        )

        logger.info(f"Created sequences: X shape {X.shape}")

        return X, y_1d, y_5d, y_20d, y_vol
    
    def create_dataloaders(self, X, y_1d, y_5d, y_20d, y_vol):
        """Create train/validation splits"""
        
        split_idx = int(0.8 * len(X))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_1d_train, y_1d_val = y_1d[:split_idx], y_1d[split_idx:]
        y_5d_train, y_5d_val = y_5d[:split_idx], y_5d[split_idx:]
        y_20d_train, y_20d_val = y_20d[:split_idx], y_20d[split_idx:]
        y_vol_train, y_vol_val = y_vol[:split_idx], y_vol[split_idx:]
        
        # Train dataset with multi-horizon targets
        train_dataset = data_utils.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_1d_train).float(),
            torch.from_numpy(y_5d_train).float(),
            torch.from_numpy(y_20d_train).float(),
            torch.from_numpy(y_vol_train).float()
        )
        
        val_dataset = data_utils.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_1d_val).float(),
            torch.from_numpy(y_5d_val).float(),
            torch.from_numpy(y_20d_val).float(),
            torch.from_numpy(y_vol_val).float()
        )
        
        # Custom collate function to create dict targets
        def collate_fn(batch):
            x, y_1d, y_5d, y_20d, y_vol = zip(*batch)
            return (torch.stack(x), {
                '1day': torch.stack(y_1d),
                '5day': torch.stack(y_5d),
                '20day': torch.stack(y_20d),
                'volatility': torch.stack(y_vol)
            })
        
        train_loader = data_utils.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = data_utils.DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self, epochs=50):
        """Full training pipeline"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}\n")
        
        # Fetch data (automatic)
        data = self.fetch_data()
        
        # Prepare
        X, y_1d, y_5d, y_20d, y_vol = self.prepare_data(data)
        
        # Create loaders
        train_loader, val_loader = self.create_dataloaders(X, y_1d, y_5d, y_20d, y_vol)
        
        # Initialize model
        self.model = DeepStockTransformer(
            input_dim=self.config['input_dim'],
            seq_length=self.config['seq_length'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout']
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
            lr=self.config['learning_rate'],
            loss_fn=multi_task_loss,
            checkpoint_path=self.config['checkpoint_path'],
            patience=self.config['patience']
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
    
    config = load_config('config.yaml')
    pipeline = TrainingPipeline(config)
    
    # Train (fetches data automatically)
    model = pipeline.train(epochs=config['epochs'])
    
    logger.info("Done!")

if __name__ == '__main__':
    main()
```

---

## 8. Configuration

`config.yaml`:

```yaml
# Data (auto-discovers all tickers from ../data/stocks and ../data/etfs)
data_path: ../data
seq_length: 60

# Model
input_dim: 5  # open, high, low, close, volume (VIX added in fetch)
d_model: 128
nhead: 8
num_layers: 4
dim_feedforward: 512
dropout: 0.15

# Training
batch_size: 16
epochs: 50
learning_rate: 5e-4
patience: 15

# Checkpoints
checkpoint_path: checkpoints/best_transformer.pth
```

`.env.example`:

```
NEWSAPI_KEY=your_newsapi_key_here
```

---

## 9. Requirements

`requirements.txt`:

```
torch==2.0.1
torchvision==0.15.1
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
yfinance==0.2.30
requests==2.31.0
textblob==0.17.1
pyyaml==6.0
python-dotenv==1.0.0
tqdm==4.65.0
matplotlib==3.7.0
```

**Note**: `tqdm` is used extensively throughout for progress bars on:
- Ticker discovery and file scanning
- Yahoo Finance data fetching and updates
- CSV loading from disk
- Data normalization and sequence creation
- Training and validation loops
- Inference and predictions

---

## 10. Usage

### Quick Start

```bash
# Setup
mkdir transformers
cd transformers

# Copy all files from sections above

# Install
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with NEWSAPI_KEY (optional)

# First training (auto-discovers all tickers in eews/data, updates them from Yahoo, trains)
python src/main.py --epochs 50

# Repeat retraining anytime (auto-discovers, auto-updates, auto-trains)
python src/main.py --epochs 20
```

### What Happens Each Run

```
You: python src/main.py --epochs 50
    ↓
Auto-discover ALL tickers from ../data/stocks/ and ../data/etfs/
    ↓
Update each ticker CSV with latest data from Yahoo
  - Read last date from CSV
  - Fetch from Yahoo from (last_date + 1) to today
  - Combine with existing data
  - Save back to CSV
    ↓
Load all updated ticker data from disk
    ↓
Combine all tickers into single DataFrame
    ↓
Normalize with StandardScaler
    ↓
Create multi-horizon sequences
    ↓
Train transformer (50 epochs, early stop if needed)
    ↓
Save model to checkpoints/best_transformer.pth
    ↓
Log all metrics to logs/training.log
    ↓
Done. All ticker CSVs now updated with latest data.
```

**Total time**: ~15-30 minutes (depends on number of tickers and Yahoo API speed)

### Features

- **Auto-discovers all tickers**: Scans ../data/stocks/ and ../data/etfs/
- **Auto-updates each ticker**: Only fetches new data since last date in CSV
- **Incremental updates**: First run may take longer, subsequent runs are fast
- **Saves back to same files**: ../data/stocks/AAPL.csv remains in same location, but updated
- **Error resilient**: If 1 ticker fails, continues with others
- **Full audit trail**: logs/training.log shows which tickers succeeded/failed

---

## 11. Key Features

✅ **Multi-horizon prediction**: 1, 5, 20-day forecasts
✅ **Sentiment scraping**: Automatic NewsAPI integration
✅ **Automatic data updates**: Fetches fresh data before every training
✅ **Multi-task learning**: 1day (50%) + 5day (25%) + 20day (15%) + volatility (10%)
✅ **Uncertainty quantification**: Monte Carlo dropout estimates confidence
✅ **Reuses existing data**: References ../data/ (5,884 stocks + 2,164 ETFs) No duplicate data, combines historical + fresh and updates each CSV to current data
✅ **Manual control**: You decide when to train/retrain
✅ **Early stopping**: Stops when validation improves
✅ **Mixed precision**: FP16 training for speed + memory
✅ **Progress tracking**: tqdm bars on all long-running operations (fetch, load, train, etc.)
✅ **Logging**: Full audit trail in logs/training.log

---

## 12. Performance Expectations

| Metric | Target |
|--------|--------|
| Training time (full dataset) | 10-15 min |
| Validation MSE (normalized) | 4-6e-9 |
| Directional accuracy | 52-55% |
| 1-day uncertainty | Useful for trading |
| Model size | ~1.5MB |
| VRAM needed | 3-4GB |

---

## 14. Troubleshooting & Notes

### Delisted/Bad Tickers

Some tickers in your CSV files may be:
- **Delisted** (company went private, merged, etc)
- **Invalid** (typos in original download)
- **Delisted ETFs** (fund closed)

These will fail during update but won't stop the process.

Example log:
```
[1847/2000] BADTICKER: Failed to fetch/update: No data found on yahoo for BADTICKER
[1848/2000] XYZ123: Failed to fetch/update: Invalid ticker XYZ123
```

**These failed tickers are simply skipped**. They won't be included in training data.

If you want to clean up bad tickers:

```python
# In fetch.py, after fetch_and_update_all_tickers()
# Option 1: Delete CSV files manually
# rm ../data/stocks/BADTICKER.csv

# Option 2: Add cleanup code
def remove_failed_tickers(self, failed_tickers):
    """Remove CSV files for failed tickers"""
    for ticker in failed_tickers:
        csv_path = self.get_csv_path(ticker)
        if os.path.exists(csv_path):
            os.remove(csv_path)
            logger.info(f"Removed {csv_path}")
```

### Yahoo API Rate Limiting

If you're fetching 2000+ tickers, Yahoo may temporarily block requests.

Solutions:
```python
# Add delay between requests in fetch.py
import time

def fetch_stock_and_update(self, ticker):
    # ... existing code ...
    time.sleep(0.1)  # 100ms delay between tickers
```

Or use a parallel approach:
```python
from concurrent.futures import ThreadPoolExecutor

def fetch_and_update_all_tickers_parallel(self, max_workers=5):
    """Fetch multiple tickers in parallel"""
    tickers = self.discover_all_tickers()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(self.fetch_stock_and_update, tickers))
    
    return tickers, sum(results), [t for t, r in zip(tickers, results) if not r]
```

### Large Dataset Handling

With 2000+ tickers, your combined DataFrame might be 5-10GB in RAM.

Monitor memory:
```bash
# Before running
free -h

# During running
watch -n 1 free -h
```

If memory is an issue, train on subset:
```python
# In main.py, limit tickers
def fetch_data(self):
    # Option 1: Sample tickers
    tickers = self.fetcher.discover_all_tickers()
    tickers = tickers[::10]  # Every 10th ticker (200 if you have 2000)
    
    combined = self.fetcher.load_tickers_from_disk(tickers)
    return combined
```

### CSV Format After Update

Your CSV files will maintain the original format:
```
Date,Open,High,Low,Close,Volume
2024-01-01,188.46,191.51,188.21,190.16,80000000
2024-01-02,189.50,190.75,187.30,188.92,65000000
```

With new rows appended:
```
2024-03-08,192.30,193.20,191.50,192.80,58000000
2024-03-09,192.50,194.10,192.20,193.45,62000000
```

### Checking Updates

After training, verify updates:

```bash
# Check last line of a ticker CSV
tail -1 ../data/stocks/AAPL.csv
# Output: 2024-03-09,192.50,194.10,192.20,193.45,62000000

# Check total rows
wc -l ../data/stocks/AAPL.csv
# Output: 2001 ../data/stocks/AAPL.csv (2000 data rows + 1 header)

# Check when it was modified
ls -l ../data/stocks/AAPL.csv
# -rw-r--r-- 1 user group 500000 Mar  9 12:30 ../data/stocks/AAPL.csv
```

### Logs

Full details in logs/training.log:

```bash
# View all discovered tickers
grep "Discovered" logs/training.log

# View update summary
grep "Update complete" logs/training.log

# View failed tickers
grep "Failed to fetch" logs/training.log

# View timing
grep -E "^2024-03-09" logs/training.log
```

1. cd to EEWS/Transformer folder
2. Copy all code from sections 3-8 above
3. Install requirements
4. Get NewsAPI key (free tier: 100 requests/day)
5. Run `python src/main.py`
6. Check logs/ and checkpoints/
7. Retrain anytime with fresh data

**That's it. Manual training with automatic data updates.**

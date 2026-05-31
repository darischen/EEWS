# EEWS - Elon Early Warning System

A production-ready machine learning system for predicting stock and ETF price movements with uncertainty quantification. Combines market data from 5,884 stocks and 2,164 ETFs with transformer-based deep learning for multi-horizon price forecasting.

## Overview

EEWS provides:
- **1-day, 5-day, and 20-day price forecasts** with confidence intervals
- **Volatility predictions** for risk assessment
- **Uncertainty quantification** via Monte Carlo dropout
- **Automatic data updates** from Yahoo Finance before each training run
- **Simple CLI interface** for quick predictions on any ticker
- **Interactive visualization** with matplotlib charts

### Quick Example

```bash
# Train the model (auto-fetches latest market data)
python Transformer/src/main.py --epochs 50

# Get a forecast for AAPL
python Transformer/src/cli.py AAPL --chart

# Check multiple stocks
for ticker in AAPL MSFT GOOGL NVDA; do
  python Transformer/src/cli.py $ticker
done
```

Output:
```
============================================================
Stock Price Forecast: AAPL
============================================================

Current Price (2026-05-31): $195.42

Horizon        Forecast        Change       Uncertainty
────────────────────────────────────────────────────────────
1-Day          $195.68        +0.13%       ± $2.15
5-Day          $197.85        +1.24%       ± $3.42
20-Day         $201.50        +3.08%       ± $5.80

Volatility: 0.0186 (1.86%)
```

## Project Structure

```
EEWS/
├── data/                                # Market data (8,048 total tickers)
│   ├── stocks/                          # 5,884 individual stock CSVs
│   │   ├── AAPL.csv, MSFT.csv, GOOGL.csv, ...
│   │   └── (A.csv through ZZZ.csv)
│   └── etfs/                            # 2,164 ETF CSVs
│       ├── SPY.csv, QQQ.csv, IWM.csv, ...
│       └── (broad range of ETF categories)
│
├── Transformer/                         # Primary model implementation
│   ├── README.md                        # Detailed Transformer documentation
│   ├── CLI_README.md                    # CLI usage guide
│   ├── CLI_GUIDE.md                     # Advanced CLI examples
│   ├── config.yaml                      # Model hyperparameters
│   ├── requirements.txt                 # Python dependencies
│   ├── .env.example                     # API keys template
│   │
│   ├── src/
│   │   ├── main.py                      # Training entry point
│   │   ├── cli.py                       # Prediction CLI interface
│   │   ├── plot.py                      # Visualization utilities
│   │   │
│   │   ├── data/
│   │   │   ├── fetch.py                 # Yahoo Finance + sentiment scraping
│   │   │   ├── loader.py                # Data loading utilities
│   │   │   ├── normalize.py             # StandardScaler + sequence creation
│   │   │   └── chunked_dataset.py       # Memory-efficient data handling
│   │   │
│   │   ├── models/
│   │   │   ├── transformer.py           # DeepStockTransformer architecture
│   │   │   └── loss.py                  # Multi-task loss functions
│   │   │
│   │   ├── training/
│   │   │   ├── train.py                 # Training loop with early stopping
│   │   │   └── checkpoint.py            # Model saving/loading
│   │   │
│   │   ├── inference/
│   │   │   └── predict.py               # Batch prediction + uncertainty
│   │   │
│   │   └── sentiment/
│   │       ├── align_sentiment.py       # Sentiment integration
│   │       └── create_sentiment_sequences.py
│   │
│   ├── checkpoints/                     # Saved model weights
│   ├── logs/                            # Training logs
│   └── data/processed/                  # Cached normalized sequences
│
└── LSTM/                                # Legacy implementation (archived)
    ├── v3.py                           # LSTM model code
    ├── GPUDetect.py                    # GPU utilities
    └── (previous experiments)
```

## Core Features

### Multi-Horizon Forecasting
Predict 3 different time horizons simultaneously:
- **1-day**: Intraday or next-day trading decisions
- **5-day**: Medium-term trends
- **20-day**: Monthly portfolio rebalancing

### Uncertainty Quantification
Uses Monte Carlo dropout to estimate prediction confidence:
- Narrow bands = high confidence
- Wide bands = high uncertainty
- Critical for risk-aware trading

### Automatic Data Updates
- Auto-discovers all 8,048 tickers from `data/` directory
- Fetches only NEW data since last run (incremental updates)
- Updates CSV files in-place
- Handles failed tickers gracefully (continues with others)

### Fast Predictions
Single-ticker forecasts in 1-2 seconds (CPU) or 0.5-1 sec (GPU).

### Interactive Visualization
4-panel matplotlib charts showing:
- Historical prices with forecast points
- Multi-horizon comparison with confidence bands
- Percent change visualization
- Detailed confidence intervals

### 💾 Memory-Efficient Training
Uses chunked datasets to keep RAM usage low even with 8,000+ tickers and millions of data points.

## Installation & Setup

### 1. Clone and Install

```bash
cd EEWS/Transformer
pip install -r requirements.txt
```

### 2. (Optional) Set Up API Keys

For sentiment analysis features:
```bash
cp .env.example .env
# Edit .env with your API keys:
# FINNHUB_API_KEY=your_key_here
# REDDIT_CLIENT_ID=your_id_here
# REDDIT_CLIENT_SECRET=your_secret_here
```

### 3. Train the Model

```bash
# First training (downloads latest data, trains, saves model)
python src/main.py --epochs 50

# Retrain later (incremental updates only, faster)
python src/main.py --epochs 20
```

## Usage

### Command-Line Forecasting

Get a quick text forecast:
```bash
python src/cli.py AAPL
```

With interactive chart:
```bash
python src/cli.py AAPL --chart
```

Save chart to file:
```bash
python src/cli.py MSFT --save forecast.png
```

Adjust uncertainty estimate (more passes = better but slower):
```bash
python src/cli.py SPY --mc 50 --chart
```

Use GPU (if available):
```bash
python src/cli.py GOOGL --device cuda --chart
```

### Batch Processing

Forecast multiple stocks:
```bash
for ticker in AAPL MSFT GOOGL AMZN NVDA; do
  python src/cli.py $ticker --save reports/${ticker}.png
done
```

See `Transformer/CLI_README.md` for complete CLI documentation.

### Training Configuration

Edit `Transformer/config.yaml` to customize:
```yaml
seq_length: 60              # Days of history per sequence
d_model: 128               # Embedding dimension
nhead: 8                   # Attention heads
num_layers: 4              # Transformer layers
batch_size: 16             # Training batch size
learning_rate: 5e-4        # Optimizer learning rate
epochs: 50                 # Training epochs
patience: 15               # Early stopping patience
```

## Model Architecture

### DeepStockTransformer

```
Input (batch, 60, 5)
    ↓
Linear Projection → d_model=128
    ↓
Positional Encoding (learnable)
    ↓
Transformer Encoder (4 layers, 8 heads, GELU activation)
    ↓
Take last token (pooling)
    ↓
Multi-head prediction:
  ├→ Head 1-day → Linear(128→64→1)
  ├→ Head 5-day → Linear(128→64→1)
  ├→ Head 20-day → Linear(128→64→1)
  └→ Head volatility → Linear(128→1)
    ↓
Returns: {1day, 5day, 20day, volatility}
```

### Training Details

- **Loss Function**: Multi-task weighted MSE
  - 1-day: 50% weight (most important)
  - 5-day: 25% weight
  - 20-day: 15% weight
  - Volatility: 10% weight (auxiliary task)

- **Optimization**: AdamW with mixed precision (FP16) for speed

- **Learning Rate**: Linear warmup (5 epochs) + cosine annealing

- **Early Stopping**: Patience of 15 epochs on validation loss

- **Data Split**: 70% train / 15% val / 15% test (temporal split)

## Data Pipeline

### Step 1: Auto-Discovery
```python
# Scans ../data/stocks/ and ../data/etfs/
# Returns all ticker names
tickers = fetcher.discover_all_tickers()  # ~8,048 tickers
```

### Step 2: Incremental Updates
```python
# For each ticker:
#   - Read last date from CSV
#   - Fetch from Yahoo from (last_date + 1) to today
#   - Combine with existing data
#   - Save back to same CSV
fetcher.fetch_and_update_all_tickers()
```

### Step 3: Load & Normalize
```python
# Load all updated CSVs
ticker_data = fetcher.load_tickers_grouped_from_disk()

# Normalize with global StandardScaler
normalizer.fit_transform(all_ohlcv_values)

# Create sequences (input: 60 days, targets: 1/5/20 days ahead)
X, y_1d, y_5d, y_20d, y_vol = create_sequences_per_ticker(data)
```

### Step 4: Train
```python
model = DeepStockTransformer(...)
model, history = train_model(
    model, train_loader, val_loader, 
    epochs=50, early_stop=True
)
```

## Performance Expectations

| Metric | Value |
|--------|-------|
| Training time (full dataset) | 10-30 minutes |
| Validation MSE (normalized) | 4-6e-3 |
| Directional accuracy (1-day) | 52-55% |
| Model size | ~1.5 MB |
| VRAM needed | 3-4 GB |
| Text forecast speed (CPU) | 1-2 sec |
| Text forecast speed (GPU) | 0.5-1 sec |
| With 10 MC passes | 3-5 sec |
| With charts | 5-8 sec |

## Common Tasks

### Update Data Only (No Training)
```python
from Transformer.src.data.fetch import DataFetcher

fetcher = DataFetcher(data_path='../data')
tickers, success, failed = fetcher.fetch_and_update_all_tickers()
print(f"Updated {success}/{len(tickers)} tickers")
```

### Load Pre-trained Model
```python
import torch
from Transformer.src.models.transformer import DeepStockTransformer

model = DeepStockTransformer(input_dim=5, seq_length=60)
model.load_state_dict(torch.load('Transformer/checkpoints/best_transformer.pth', weights_only=True))
model.eval()
```

### Make Predictions Programmatically
```python
from Transformer.src.inference.predict import predict_single_ticker

predictions, uncertainties = predict_single_ticker(
    model, X_tensor, device, num_passes=10
)
print(f"1-day: {predictions['1day']:.2f} ± {uncertainties['1day']:.2f}")
```

## Troubleshooting

### "Model checkpoint not found"
Train the model first:
```bash
python src/main.py --epochs 50
```

### "Ticker not found"
The ticker CSV doesn't exist in `data/stocks/` or `data/etfs/`. Run training to fetch it:
```bash
python src/main.py --epochs 20
```

### Out of memory during training
Reduce batch size in `config.yaml`:
```yaml
batch_size: 8  # Lower from 16
```

Or train on a subset of tickers:
```python
# In src/main.py, modify fetch_data():
tickers = self.fetcher.discover_all_tickers()[::10]  # Every 10th ticker
```

### Slow Yahoo Finance downloads
Yahoo may rate-limit requests. The code auto-retries, but you can add delays:
```python
# In src/data/fetch.py, add after yf.download():
import time
time.sleep(0.1)  # 100ms between tickers
```

## Recent Updates

### May 2026
- **CSV Data Normalization**: Standardized all 8,048 stock/ETF CSV headers to lowercase
  - `Date` → `date`, `Open` → `open`, `High` → `high`, `Low` → `low`, `Close` → `close`, `Volume` → `volume`
  
- **Transformer Model**: Complete implementation with:
  - Multi-horizon prediction (1, 5, 20-day)
  - Monte Carlo dropout for uncertainty
  - Multi-task learning with volatility prediction
  - Sentiment analysis integration (optional)
  - Automatic data fetching and updates
  - Interactive CLI with visualization

- **Documentation**: Comprehensive guides for training and prediction

## File Size Reference

| Component | Size |
|-----------|------|
| Data CSVs (all 8,048) | ~10-15 GB |
| Trained model checkpoint | ~1.5 MB |
| Training log | ~5-10 MB |
| Full project (no data) | ~50 MB |

## Future Improvements

- [ ] Ensemble methods combining LSTM + Transformer
- [ ] Real-time prediction server with FastAPI
- [ ] Backtesting framework with Zipline
- [ ] Portfolio optimization using predictions
- [ ] Advanced sentiment analysis (FinBERT)
- [ ] Reinforcement learning for trading strategy

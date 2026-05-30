# Stock Price Forecast CLI

Simple command-line interface to generate stock price forecasts with matplotlib charts.

## Quick Start

```bash
# Basic forecast (text only)
python src/cli.py AAPL

# With charts
python src/cli.py AAPL --chart

# Save chart to file
python src/cli.py MSFT --save forecast.png

# Adjust uncertainty estimation
python src/cli.py SPY --mc 20  # Use 20 MC dropout passes (default: 10)
```

## Usage

```
python src/cli.py TICKER [options]

Positional Arguments:
  TICKER              Stock ticker symbol (e.g., AAPL, MSFT, SPY)

Optional Arguments:
  --chart             Display matplotlib chart
  --save FILE         Save chart to file (e.g., forecast.png)
  --mc PASSES         Number of MC dropout passes for uncertainty (default: 10)
  --device DEVICE     Compute device: cpu or cuda (default: cpu)
  -h, --help          Show help message
```

## Examples

### 1. Text-only forecast

```bash
$ python src/cli.py AAPL

============================================================
Stock Price Forecast: AAPL
============================================================

✓ Loaded model from checkpoints/best_transformer.pth
✓ Loaded 880 rows of AAPL data
  Date range: 2023-01-01 to 2026-03-13
✓ Normalized data (mean=0, std=1)

────────────────────────────────────────────────────────────
Generating Forecast (MC passes: 10)
────────────────────────────────────────────────────────────

Current Price (2026-03-13): $195.42

Horizon        Forecast        Change       Uncertainty
────────────────────────────────────────────────────────────
1-Day          $195.68        +0.13%       ± $2.15
5-Day          $197.85        +1.24%       ± $3.42
20-Day         $201.50        +3.08%       ± $5.80

Volatility: 0.0186 (1.86%)

============================================================
```

### 2. With chart

```bash
python src/cli.py AAPL --chart
```

Creates a 2x2 grid of plots:
- **Top-left**: Historical data with 20-day forecast point
- **Top-right**: Multi-horizon price comparison with uncertainty bars
- **Bottom-left**: Percentage change from current price
- **Bottom-right**: Detailed confidence intervals

### 3. Save chart to file

```bash
python src/cli.py MSFT --save msft_forecast.png

# Generates a high-resolution PNG (150 dpi)
ls -lh msft_forecast.png
# -rw-r--r-- 1 user group 245K Mar 13 14:30 msft_forecast.png
```

### 4. Higher uncertainty precision

```bash
python src/cli.py GOOGL --mc 50 --chart

# Uses 50 MC dropout passes (slower, more accurate uncertainty)
# Default is 10 passes (fast, reasonable accuracy)
```

### 5. GPU acceleration (if available)

```bash
python src/cli.py TSLA --device cuda --chart

# Falls back to CPU if CUDA unavailable
```

## Output Interpretation

### Price Forecasts

- **1-Day**: Price tomorrow (for day-trading decisions)
- **5-Day**: Price in 5 days (short-term trend)
- **20-Day**: Price in 20 days (medium-term outlook)

### Uncertainty

- **± Value**: Prediction confidence band (68% confidence interval)
- Larger uncertainty = less confident prediction
- Smaller uncertainty = more confident prediction

### Volatility

- Estimated price volatility over the prediction horizon
- 0.018 = 1.8% expected volatility
- Higher volatility = more risk/opportunity

### Percent Change

- **+1.24%**: Expected to gain 1.24% in 5 days
- **-2.30%**: Expected to lose 2.30% in 20 days
- Helps compare to benchmark returns

## Chart Panels

### Panel 1: Historical Data & 20-Day Forecast
- Green dot: Current price
- Blue line: Last 3 months of historical data
- Red square: 20-day forecast
- Red band: Confidence interval

### Panel 2: Multi-Horizon Comparison
- Bar chart showing 1-day, 5-day, 20-day forecasts
- Error bars show uncertainty ranges
- Easy visual comparison

### Panel 3: Percent Change
- Green bars: Expected price increase
- Red bars: Expected price decrease
- Shows relative movements at a glance

### Panel 4: Confidence Intervals
- Center bar: Mean prediction
- Red dashed lines: Uncertainty bounds
- Width indicates model confidence

## Prerequisites

Before using CLI, you must:

1. Train the model (takes ~15-30 minutes):
```bash
python src/main.py --epochs 50
```

2. The data must be updated:
```bash
# main.py automatically fetches latest data
# But make sure ../data/stocks/ and ../data/etfs/ have CSV files
ls ../data/stocks/AAPL.csv  # Should exist
```

## Troubleshooting

### "Model checkpoint not found"

```
❌ Error: Model checkpoint not found: checkpoints/best_transformer.pth
Please run: python src/main.py --epochs 50
```

**Solution**: Train the model first
```bash
python src/main.py --epochs 50
```

### "Ticker not found"

```
❌ Error: Ticker 'XYZ' not found in ../data/stocks or /etfs
Run src/main.py first to fetch and update data.
```

**Solution**: Either:
1. The ticker doesn't exist in your data
2. Run `src/main.py` to update all tickers

### "Not enough data"

```
❌ Error: Not enough data for TICKER. Need at least 60 days, got 10
```

**Solution**: The ticker has less than 60 days of history. It will be available after the next `src/main.py` run.

### Chart not displaying

If `--chart` doesn't show the plot:
- On headless servers: Use `--save file.png` instead
- On Windows with WSL: Configure display or use `--save`
- Check if matplotlib backend is available: `python -c "import matplotlib; print(matplotlib.get_backend())"`

## Performance

- **Text forecast**: ~1-2 seconds per ticker
- **With 10 MC passes**: ~3-5 seconds
- **With chart generation**: +2-3 seconds
- **GPU (CUDA)**: 2-3x faster

## Tips

### For Traders
```bash
# Quick morning check
for ticker in AAPL MSFT GOOGL; do
  python src/cli.py $ticker
  echo "---"
done
```

### For Analysis
```bash
# Save charts for multiple tickers
for ticker in SPY QQQ IWM; do
  python src/cli.py $ticker --mc 50 --save ${ticker}_forecast.png
done

# View all results
ls -lh *.png
open *.png  # macOS
# or
feh *.png   # Linux
```

### For Batch Processing
```bash
# Get forecasts as CSV (coming soon)
python scripts/forecast_batch.py tickers.csv > forecasts.csv
```

## Limitations

- Model trained on historical data - past performance ≠ future results
- Short-term predictions (20 days max) - use for tactical, not strategic decisions
- Uncertainty estimates improve with more data
- Works best for liquid, frequently-traded stocks

## Next Steps

- [Training Guide](Claude.md) - How to retrain with new data
- [Architecture Details](Claude.md#4-deeptransformer) - Model internals
- [Data Updates](Claude.md#3-data-fetching) - How data is fetched and updated

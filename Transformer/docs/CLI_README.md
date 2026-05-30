# Stock Price Forecast CLI ✨

A simple command-line interface to generate 1-day, 5-day, and 20-day stock price forecasts with interactive matplotlib charts.

## 🚀 Quick Start

```bash
# 1. Train the model (one-time, ~20 minutes)
python src/main.py --epochs 50

# 2. Get a forecast for any stock
python src/cli.py AAPL

# 3. With charts
python src/cli.py AAPL --chart

# 4. Save chart to file
python src/cli.py MSFT --save forecast.png
```

## 📊 What You Get

```
python src/cli.py AAPL

============================================================
Stock Price Forecast: AAPL
============================================================

Current Price (2026-03-13): $195.42

Horizon        Forecast        Change       Uncertainty
────────────────────────────────────────────────────────────
1-Day          $195.68        +0.13%       ± $2.15
5-Day          $197.85        +1.24%       ± $3.42
20-Day         $201.50        +3.08%       ± $5.80

Volatility: 0.0186 (1.86%)
```

Plus beautiful matplotlib charts showing:
- Historical data with forecast points
- Multi-horizon comparison with uncertainty bars
- Percent change visualization
- Detailed confidence intervals

## 📁 Files Created

```
Transformer/
├── src/
│   ├── __init__.py                 # Module init
│   ├── cli.py                      # 📌 Main CLI interface (NEW)
│   ├── plot.py                     # 📌 Matplotlib visualization (NEW)
│   └── inference/
│       ├── __init__.py
│       └── predict.py              # 📌 Updated prediction functions (NEW)
├── CLI_GUIDE.md                    # Detailed usage guide
├── CLI_README.md                   # This file
├── examples.sh                     # Example commands
└── requirements.txt                # Updated with matplotlib
```

## 🛠️ Installation

```bash
# Install dependencies (matplotlib already included)
pip install -r requirements.txt
```

## 💻 Usage

### Basic Forecast (Text)
```bash
python src/cli.py AAPL
```

### With Interactive Chart
```bash
python src/cli.py AAPL --chart
```

### Save Chart to File
```bash
python src/cli.py AAPL --save my_chart.png
```

### Adjust Precision
```bash
# More MC dropout passes = better uncertainty estimate (slower)
python src/cli.py AAPL --mc 50 --chart

# Default is 10 passes (fast, good enough)
python src/cli.py AAPL --mc 10
```

### Use GPU (if available)
```bash
python src/cli.py AAPL --device cuda --chart
```

### All Options
```bash
python src/cli.py --help

usage: cli.py [-h] [--chart] [--save SAVE] [--mc MC] [--device DEVICE] ticker

Stock price forecast using Transformer model

positional arguments:
  ticker           Stock ticker symbol (e.g., AAPL, MSFT, SPY)

optional arguments:
  -h, --help       show this help message and exit
  --chart          Display matplotlib chart
  --save SAVE      Save chart to file (e.g., forecast.png)
  --mc MC          Number of MC dropout passes for uncertainty (default: 10)
  --device DEVICE  Compute device: cpu or cuda (default: cpu)
```

## 📈 Chart Panels

### Panel 1: Historical Data & 20-Day Forecast
- Blue line: Last 3 months of historical prices
- Green dot: Current price
- Red square: 20-day forecast
- Red band: Confidence interval

### Panel 2: Multi-Horizon Comparison
- Bar chart of 1-day, 5-day, 20-day forecasts
- Error bars show prediction uncertainty
- Easy visual comparison

### Panel 3: Percent Change
- Green bars: Expected gains
- Red bars: Expected losses
- Shows relative movements

### Panel 4: Confidence Intervals
- Center bar: Mean prediction
- Red dashed lines: ±Uncertainty bounds
- Wider bands = less confident

## 🎯 Use Cases

### Day Trader
```bash
# Quick morning check of key stocks
python src/cli.py SPY --mc 20
python src/cli.py QQQ --mc 20
python src/cli.py IWM --mc 20
```

### Analysis
```bash
# Generate charts for report
mkdir reports
for ticker in AAPL MSFT GOOGL AMZN NVDA; do
  python src/cli.py $ticker --save reports/${ticker}.png
done
```

### Batch Processing
```bash
# Check 10 stocks, no charts (faster)
for ticker in $(cat watchlist.txt); do
  python src/cli.py $ticker
done
```

## ⚠️ Prerequisites

1. **Model must be trained** - Run once:
   ```bash
   python src/main.py --epochs 50
   ```
   Creates: `checkpoints/best_transformer.pth`

2. **Data must exist** - The model needs:
   - `../data/stocks/*.csv` (at least one stock CSV)
   - `../data/etfs/*.csv` (at least one ETF CSV)
   - Run `src/main.py` to fetch/update

## 📊 Understanding Output

### Price Forecasts
- **$195.68**: Predicted close price
- **±$2.15**: Confidence range (68% confidence)
- Narrower = more confident

### Percent Change
- **+1.24%**: Expect 1.24% gain in 5 days
- **-0.50%**: Expect 0.50% loss in 1 day

### Volatility
- **0.0186**: 1.86% expected daily volatility
- Used to estimate trading ranges

## ⚡ Performance

| Scenario | Time |
|----------|------|
| Text forecast (cpu) | 1-2 sec |
| Text forecast (gpu) | 0.5-1 sec |
| With 10 MC passes | 3-5 sec |
| With chart + 10 MC passes | 5-8 sec |
| Save chart (no display) | 7-10 sec |

## 🐛 Troubleshooting

### "Model checkpoint not found"
```
Train the model first:
python src/main.py --epochs 50
```

### "Ticker not found"
```
The ticker CSV doesn't exist. Run main.py to fetch data:
python src/main.py
```

### Chart not showing on Linux/WSL
```
Use --save instead:
python src/cli.py AAPL --save forecast.png
```

## 📚 More Info

- **Full guide**: `CLI_GUIDE.md`
- **Examples**: `examples.sh`
- **Architecture**: `Claude.md`
- **Training**: `Claude.md#5-training-loop`

## 🎓 Examples

See `examples.sh` for copy-paste ready examples:
```bash
bash examples.sh
```

## 🚀 Next Steps

1. Train the model (first time only)
2. Run `python src/cli.py AAPL --chart` to test
3. Check `CLI_GUIDE.md` for advanced usage
4. Create a watchlist and check multiple stocks

---

**Built with PyTorch Transformers | Data from Yahoo Finance & Sentiment (Finnhub + Reddit)**

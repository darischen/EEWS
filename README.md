# Elon Early Warning System (EEWS)

Currently implementing a stock prediction LSTM RNN model using cuda and torch

## Features implemented
- Trained on historical data of the majority of stocks
  - Data ranges from the company's inception until 4/1/2020
- The dataset is sourced from Yahoo Finance
- The model is trained on the opening and closing price, trade volume, the high, low, and adj close of each day.
- The model uses the last 20 days to make predictions
- Some training session with notably low losses have been saved in the .pth files, with the names denoting the loss.

## Future implementations
- The model will take into account
  - Bollinger bands
  - Stochastic Oscillator
  - Exponential Moving Average (EMA)
  - Simple Moving Average (SMA)
  - Volatility Index (VIX)
  - Moving Average Convergence Divergence (MACD)

## Running locally
### Create conda environment and activate it
```
conda create -n eews python=3.10  
conda activate eews
```
### Install dependencies
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
### Run the model
- Either use the jupyter notebooks or run the script directly using
```python v3.py```
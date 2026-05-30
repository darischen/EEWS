"""
Plotting utilities for forecast visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta


def plot_forecast(ticker, df, current_price, pred_1day, pred_5day, pred_20day,
                  unc_1day, unc_5day, unc_20day, show=True, save_path=None):
    """
    Create multi-panel forecast visualization

    Args:
        ticker: Stock ticker
        df: Historical OHLCV DataFrame
        current_price: Current close price
        pred_*: Predicted prices
        unc_*: Prediction uncertainties
        show: Display plot
        save_path: Save to file
    """

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'{ticker} Stock Price Forecast', fontsize=18, fontweight='bold', y=0.995)

    # ─────────────────────────────────────────────────────────────
    # Panel 1: Historical data + 20-day forecast
    # ─────────────────────────────────────────────────────────────
    ax1 = plt.subplot(2, 2, 1)

    # Get last 3 months for context
    cutoff = max(0, len(df) - 90)
    df_recent = df.iloc[cutoff:]

    ax1.plot(df_recent.index, df_recent['close'], 'o-', linewidth=2.5,
             markersize=3, label='Historical', color='#1f77b4')

    # Add current price marker
    current_date = df.index[-1]
    ax1.plot(current_date, current_price, 'o', markersize=10,
             color='green', label=f'Current (${current_price:.2f})', zorder=5)

    # Add 20-day forecast point
    forecast_date = current_date + timedelta(days=20)
    ax1.plot(forecast_date, pred_20day, 's', markersize=10,
             color='red', label=f'20-Day Forecast (${pred_20day:.2f})', zorder=5)

    # Uncertainty band for 20-day
    ax1.fill_between(
        [current_date, forecast_date],
        [current_price, pred_20day - unc_20day],
        [current_price, pred_20day + unc_20day],
        alpha=0.2, color='red', label=f'Confidence: ±${unc_20day:.2f}'
    )

    ax1.set_title('Historical Data & 20-Day Forecast', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # ─────────────────────────────────────────────────────────────
    # Panel 2: Multi-horizon forecast comparison
    # ─────────────────────────────────────────────────────────────
    ax2 = plt.subplot(2, 2, 2)

    horizons = ['Current', '1-Day', '5-Day', '20-Day']
    prices = [current_price, pred_1day, pred_5day, pred_20day]
    uncertainties = [0, unc_1day, unc_5day, unc_20day]
    colors = ['green', '#ff7f0e', '#2ca02c', 'red']

    bars = ax2.bar(horizons, prices, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add error bars
    ax2.errorbar(horizons, prices, yerr=uncertainties, fmt='none',
                 ecolor='black', capsize=8, capthick=2, linewidth=2, alpha=0.6)

    # Add value labels on bars
    for bar, price in zip(bars, prices):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${price:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax2.set_ylabel('Price ($)', fontsize=11)
    ax2.set_title('Multi-Horizon Price Forecast', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(min(prices) * 0.95, max(prices) * 1.05)

    # ─────────────────────────────────────────────────────────────
    # Panel 3: Percentage change from current
    # ─────────────────────────────────────────────────────────────
    ax3 = plt.subplot(2, 2, 3)

    pct_changes = [
        0,
        ((pred_1day - current_price) / current_price) * 100,
        ((pred_5day - current_price) / current_price) * 100,
        ((pred_20day - current_price) / current_price) * 100,
    ]

    colors_change = ['gray' if x == 0 else ('green' if x > 0 else 'red') for x in pct_changes]
    bars = ax3.bar(horizons, pct_changes, color=colors_change, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, pct in zip(bars, pct_changes):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = height * 0.02 if height >= 0 else height * 0.02
        ax3.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{pct:+.2f}%',
                ha='center', va=va, fontweight='bold', fontsize=10)

    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Change (%)', fontsize=11)
    ax3.set_title('Percent Change from Current Price', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # ─────────────────────────────────────────────────────────────
    # Panel 4: Uncertainty ranges
    # ─────────────────────────────────────────────────────────────
    ax4 = plt.subplot(2, 2, 4)

    forecast_horizons = ['1-Day', '5-Day', '20-Day']
    forecast_prices = [pred_1day, pred_5day, pred_20day]
    forecast_uncs = [unc_1day, unc_5day, unc_20day]

    x_pos = np.arange(len(forecast_horizons))

    # Plot mean
    ax4.bar(x_pos, forecast_prices, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add error bars for uncertainty
    ax4.errorbar(x_pos, forecast_prices, yerr=forecast_uncs, fmt='none',
                 ecolor='red', capsize=10, capthick=2.5, linewidth=2.5, alpha=0.8, label='±Uncertainty')

    # Add confidence interval bands
    for i, (price, unc) in enumerate(zip(forecast_prices, forecast_uncs)):
        ax4.plot([i-0.3, i+0.3], [price-unc, price-unc], 'r--', linewidth=2, alpha=0.6)
        ax4.plot([i-0.3, i+0.3], [price+unc, price+unc], 'r--', linewidth=2, alpha=0.6)

    # Add value labels
    for i, (price, unc) in enumerate(zip(forecast_prices, forecast_uncs)):
        ax4.text(i, price + unc + 1, f'${price:.2f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9)
        ax4.text(i, price - unc - 1, f'±${unc:.2f}', ha='center', va='top',
                fontsize=8, style='italic', color='red')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(forecast_horizons)
    ax4.set_ylabel('Price ($)', fontsize=11)
    ax4.set_title('Prediction Uncertainty (Confidence Intervals)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(loc='best', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_simple_forecast(ticker, df, current_price, pred_1day, pred_5day,
                         pred_20day, show=True, save_path=None):
    """
    Simpler single-panel forecast visualization
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical data
    cutoff = max(0, len(df) - 60)
    df_recent = df.iloc[cutoff:]

    ax.plot(df_recent.index, df_recent['close'], 'o-', linewidth=2.5,
            markersize=4, label='Historical Close', color='#1f77b4')

    # Add current price marker
    current_date = df.index[-1]
    ax.plot(current_date, current_price, 'o', markersize=12,
            color='green', label=f'Current (${current_price:.2f})', zorder=5)

    # Add forecast points
    dates = [
        current_date + timedelta(days=1),
        current_date + timedelta(days=5),
        current_date + timedelta(days=20),
    ]
    preds = [pred_1day, pred_5day, pred_20day]
    labels = ['1-Day', '5-Day', '20-Day']
    colors = ['#ff7f0e', '#2ca02c', 'red']

    for date, pred, label, color in zip(dates, preds, labels, colors):
        ax.plot(date, pred, 's', markersize=10, color=color, label=f'{label} (${pred:.2f})', zorder=5)

    ax.set_title(f'{ticker} Stock Price Forecast', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

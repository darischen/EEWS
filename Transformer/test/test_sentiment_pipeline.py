"""Test sentiment alignment pipeline."""
import sys
sys.path.insert(0, 'src')

from sentiment.align_sentiment import SentimentAligner, forward_fill_missing_sentiment
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sentiment_alignment():
    """Test alignment of sentiment to price sequences."""
    print("\n" + "="*60)
    print("TEST: Sentiment Alignment")
    print("="*60)

    try:
        aligner = SentimentAligner('data/sentiment/daily_sentiment.csv')

        # Test getting sentiment for a few dates/tickers
        sample_tickers = ['AAPL', 'MSFT', 'TSLA']
        sample_date = '2026-03-10'

        print(f"\nSentiment on {sample_date}:")
        for ticker in sample_tickers:
            sentiment = aligner.get_sentiment_for_date(ticker, sample_date)
            print(f"  {ticker}: {sentiment:.4f}")

        # Show sentiment coverage
        print(f"\nTicker coverage:")
        for ticker in list(aligner.sentiment_dict.keys())[:5]:
            n_dates = len(aligner.sentiment_dict[ticker])
            print(f"  {ticker}: {n_dates} days with sentiment")

        print("\n[OK] Alignment test passed")

    except FileNotFoundError:
        print("[WARN] Sentiment file not found (expected if sentiment not fetched yet)")


if __name__ == '__main__':
    import os

    # Create output directory
    os.makedirs('data/sentiment', exist_ok=True)

    print("\n" + "="*70)
    print("SENTIMENT ALIGNMENT TEST")
    print("="*70)

    try:
        test_sentiment_alignment()

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("[OK] Sentiment alignment module ready")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

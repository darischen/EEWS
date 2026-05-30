#!/bin/bash
# Stock Price Forecast CLI - Common Examples

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Stock Price Forecast CLI - Usage Examples              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Example 1: Basic forecast
echo -e "${GREEN}1. Basic text forecast (no chart):${NC}"
echo "   python src/cli.py AAPL"
echo ""

# Example 2: With chart
echo -e "${GREEN}2. Forecast with interactive chart:${NC}"
echo "   python src/cli.py MSFT --chart"
echo ""

# Example 3: Save chart
echo -e "${GREEN}3. Save chart to file:${NC}"
echo "   python src/cli.py GOOGL --save googl_forecast.png"
echo ""

# Example 4: High precision
echo -e "${GREEN}4. Higher uncertainty precision (slower):${NC}"
echo "   python src/cli.py TSLA --mc 50 --chart"
echo ""

# Example 5: Multiple tickers
echo -e "${GREEN}5. Quick check multiple tickers:${NC}"
echo "   for ticker in AAPL MSFT GOOGL AMZN NVDA; do"
echo "     echo \"─────────────────────────────────────\""
echo "     python src/cli.py \$ticker"
echo "   done"
echo ""

# Example 6: Batch save charts
echo -e "${GREEN}6. Save charts for multiple tickers:${NC}"
echo "   mkdir -p charts"
echo "   for ticker in SPY QQQ IWM; do"
echo "     python src/cli.py \$ticker --save charts/\${ticker}_forecast.png"
echo "   done"
echo "   ls -lh charts/"
echo ""

# Example 7: GPU acceleration
echo -e "${GREEN}7. Use GPU if available (faster):${NC}"
echo "   python src/cli.py NVDA --device cuda --chart"
echo ""

# Example 8: Check setup
echo -e "${GREEN}8. Verify model is trained:${NC}"
echo "   ls -lh checkpoints/best_transformer.pth"
echo ""

echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
echo -e "${GREEN}Before running: Train the model${NC}"
echo "   python src/main.py --epochs 50"
echo ""
echo -e "${GREEN}View full documentation:${NC}"
echo "   cat CLI_GUIDE.md"
echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
echo ""

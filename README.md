# EEWS - Energy & Economic Wealth Signals

A machine learning project for predicting financial market movements using deep learning models.

## Project Structure

- **data/** - Market data and financial datasets
  - `etfs/` - Exchange-traded fund (ETF) price data (5000+ files)
  - `stocks/` - Individual stock price data
  - `*.csv` - Various market data files (AAPL, AMZN, TSLA, etc.)

- **LSTM/** - Long Short-Term Memory model implementations
  - Model checkpoints and weights
  - Training notebooks and scripts
  - Performance metrics and evaluation data

- **Transformer/** - Transformer-based model implementations
  - Configuration files for model training
  - Source code and utilities
  - Test suites and documentation

## Recent Changes

### May 2026
- **CSV Data Normalization**: Standardized all ETF and stock CSV headers to lowercase column names
  - Changed headers: `Date` → `date`, `Open` → `open`, `High` → `high`, `Low` → `low`, `Close` → `close`, `Adj Close` → `adj close`, `Volume` → `volume`
  - Affects 2000+ data files ensuring consistent column naming across the project

- **Project Configuration**: 
  - Updated `.gitignore` to exclude `.claude/` directory (local IDE settings)
  - Configured Git permissions for automated deployments

- **Documentation Cleanup**: Removed redundant Transformer module documentation

## Dependencies

See `requirements.txt` in both LSTM and Transformer directories for model-specific dependencies.

## Models

### LSTM Models
Traditional recurrent neural network approach for time series prediction of market movements.

### Transformer Models
Modern attention-based architecture for improved temporal pattern recognition in financial data.

## Usage

Install dependencies and explore the Jupyter notebooks in each model directory for training and evaluation examples.

## License

Educational project

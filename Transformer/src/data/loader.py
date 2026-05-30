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

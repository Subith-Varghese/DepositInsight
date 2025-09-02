import pandas as pd
import os
from src.logger import logger

class DataIngestion:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Load the CSV dataset
        """
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found at {self.file_path}")
            data = pd.read_csv(self.file_path, sep=';')
            logger.info(f"Dataset loaded successfully with shape {data.shape}")
            return data
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            raise e

    def separate_null_target(self, data, target='y'):
        """
        Separate rows with null target values (for later prediction)
        """
        test_data = data[data[target].isnull()].copy()
        data = data.dropna(subset=[target])
        logger.info(f"Separated {len(test_data)} rows with null target values")
        return data, test_data

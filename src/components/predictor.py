import os
import pickle
import joblib
import pandas as pd
from src.logger import logger

ARTIFACTS_DIR = "artifacts"

# Columns dropped during training
DROPPED_COLUMNS = [
    "housing", "loan",              
    "nr.employed", "cons.price.idx",
    "euribor3m", "cons.conf.idx",
    "pdays"                          
]

class Predictor:
    def __init__(self, model_name="random_forest"):
        """
        model_name: filename prefix of the trained model
        """
        self.model_file = os.path.join(ARTIFACTS_DIR, f"{model_name}.pkl")
        self.encoders_file = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
        self.scaler_file = os.path.join(ARTIFACTS_DIR, "scaler.pkl")


        # Load model
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file not found at {self.model_file}")
        self.model = joblib.load(self.model_file)
        logger.info(f"Loaded model from {self.model_file}")

        # Load encoders
        if os.path.exists(self.encoders_file):
            with open(self.encoders_file, "rb") as f:
                self.encoders = pickle.load(f)
            logger.info("Loaded label encoders")
        else:
            self.encoders = {}

        # Load scaler
        if os.path.exists(self.scaler_file):
            with open(self.scaler_file, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded StandardScaler")
        else:
            self.scaler = None

    def preprocess(self, df):
        # Drop same columns as training
        for col in DROPPED_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Encode categorical features
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        
        # Group age like in training
        if 'age' in df.columns:
            df['age'] = df['age'] // 10 * 10
        
        return df

    def predict(self, df):
        preprocessed = self.preprocess(df)
        preprocessed = self.scaler.transform(preprocessed)
        preds = self.model.predict(preprocessed)
        logger.info(f"Predictions made for {len(preds)} rows")
        return preds

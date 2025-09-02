import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.logger import logger
import os
import pickle


ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

class DataTransformation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}

    def handle_missing_values(self, data):
        for col in data.columns:
            if data[col].dtype == 'float64':
                median_val = round(data[col].median())
                data[col] = data[col].fillna(median_val)
            elif data[col].dtype == 'object':
                data[col] = data[col].fillna(data[col].mode()[0])
        logger.info("Missing values handled")
        return data
    
    def remove_outliers(self, data, columns=None):
        """
        Remove outliers using IQR method for the given list of columns
        """
        if columns is None:
            columns = ['age', 'duration', 'campaign', 'cons.conf.idx']

        for col in columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            original_len = len(data)
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            logger.info(f"Removed {original_len - len(data)} outliers from column '{col}'")
        return data

    def group_age(self, data):
        if 'age' in data.columns:
            data['age'] = data['age']//10*10
        return data

    def encode_categorical(self, data, target='y'):
        cat_cols = data.select_dtypes(include='object').columns.tolist()
        if target in cat_cols:
            cat_cols.remove(target)

        for col in cat_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.encoders[col] = le
        logger.info(f"Encoded categorical columns: {list(self.encoders.keys())}")
        
        encoder_file = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
        with open(encoder_file, "wb") as f:
            pickle.dump(self.encoders,f)   
        return data

    def remove_multicollinearity(self, data, target='y', threshold=10.0):
        x = data.drop(columns=[target])

        while True:
            vif = pd.DataFrame()
            vif['features'] = x.columns
            vif['vif'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

            max_vif = vif['vif'].max()
            if max_vif > threshold:
                drop_feature = vif.loc[vif['vif'].idxmax(), 'features']
                x = x.drop(drop_feature, axis=1)
                data = data.drop(drop_feature, axis=1)
                logger.info(f"Dropped {drop_feature} with VIF={max_vif:.2f}")
            else:
                break
        return data

    def scale_features(self, x_train, x_test):
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        logger.info("Features scaled")
        scaler_file = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump(self.scaler,f) 
        return x_train_scaled, x_test_scaled

# src/components/feature_selection.py

import pandas as pd
from scipy.stats import chi2_contingency, chi2
from src.logger import logger

class FeatureSelection:
    def __init__(self, categorical_columns=None, target='y', alpha=0.05):
        """
        categorical_columns: list of categorical columns to test
        target: target variable name
        alpha: significance level
        """
        self.categorical_columns = categorical_columns
        self.target = target
        self.alpha = alpha
        self.dropped_features = []

    def chi_square_test(self, data):
        """
        Performs Chi-square test for all categorical features
        Drops features with p-value >= alpha
        Returns: data with insignificant features removed
        """
        if self.categorical_columns is None:
            self.categorical_columns = data.select_dtypes(include='object').columns.tolist()
            if self.target in self.categorical_columns:
                self.categorical_columns.remove(self.target)

        for col in self.categorical_columns:
            contingency_table = pd.crosstab(data[col], data[self.target])
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            if p_value >= self.alpha:
                data = data.drop(columns=[col])
                self.dropped_features.append(col)
                logger.info(f"Dropped '{col}' due to Chi-square test (p-value={p_value:.4f})")

        return data

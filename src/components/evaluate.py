import os
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from src.components.predictor import Predictor
from src.logger import logger
from sklearn.model_selection import train_test_split
import joblib



def evaluate_model(model_name, X_test, y_test):
    """
    Evaluates a single model on the test set and prints metrics.
    """
    logger.info(f"=== Evaluating Model: {model_name} ===")
    
    # Load predictor
    predictor = Predictor(model_name=model_name)

    # Generate predictions
    y_pred = predictor.predict(X_test)

    # Print confusion matrix and classification report
    logger.info(f"Evaluate result for {model_name}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    logger.info("======================================================\n")


models =[
    "decision_tree",
    "logistic_regression",
    "random_forest",
    "random_forest_best"]

if __name__ == "__main__":
    data_path =  "data/bank-additional-full.csv"
    df = pd.read_csv(data_path, sep=";")

    X = df.drop(columns=['y'])
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for model in models:
        evaluate_model(model,X_test,y_test)
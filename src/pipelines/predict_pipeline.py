import os
import pandas as pd
from src.components.predictor import Predictor
from src.logger import logger

# Path to your test dataset
DATA_PATH = "data/new_customers.csv"
ARTIFACTS_DIR = "artifacts"


def run_prediction_pipeline(model_name="random_forest"):
    """
    Runs the prediction pipeline for unseen data without 'y' column.
    """
    logger.info("========== PREDICTION PIPELINE STARTED ==========")

    # 1️⃣ Load dataset directly (since this is new unseen data)
    try:
        test_data = pd.read_csv(DATA_PATH, sep=";")
        logger.info(f"Dataset loaded successfully with shape {test_data.shape}")
    except Exception as e:
        logger.exception(f"Error while loading dataset: {e}")
        raise e

    # 2️⃣ Check if dataset is empty
    if test_data.empty:
        logger.warning("No data available for prediction!")
        return

    # 3️⃣ Drop target column if accidentally present
    if 'y' in test_data.columns:
        test_features = test_data.drop(columns=['y'])
    else:
        test_features = test_data.copy()

    # 4️⃣ Initialize predictor with trained model
    predictor = Predictor(model_name=model_name)

    # 5️⃣ Make predictions
    predictions = predictor.predict(test_features)

    # 6️⃣ Add predictions to dataframe
    test_data['predicted_y'] = predictions

    # 7️⃣ Save predictions into artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    output_file = os.path.join(ARTIFACTS_DIR, "predictions.csv")
    test_data.to_csv(output_file, index=False)

    logger.info(f"Predictions saved successfully at {output_file}")
    logger.info("========== PREDICTION PIPELINE COMPLETED ==========")

    # Show sample predictions
    print(test_data.head())


if __name__ == "__main__":
    # You can choose any model: logistic_regression, decision_tree, random_forest
    run_prediction_pipeline(model_name="random_forest_best")

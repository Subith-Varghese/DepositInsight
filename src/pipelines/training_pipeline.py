from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.feature_selection import FeatureSelection
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from src.logger import logger

DATA_PATH = "data/bank-additional-full.csv"
ARTIFACTS_DIR = "artifacts"

def run_training_pipeline():
    # 1️⃣ Data ingestion
    ingestion = DataIngestion(DATA_PATH)
    data = ingestion.load_data()
    data,_ = ingestion.separate_null_target(data)

    # 2️⃣ Data transformation
    transformer = DataTransformation()
    data = transformer.handle_missing_values(data)
    data = transformer.group_age(data)

    selector = FeatureSelection()
    data = selector.chi_square_test(data)

    data = transformer.encode_categorical(data)
    data = transformer.remove_multicollinearity(data)

    # 3️⃣ Feature split
    X = data.drop(columns=['y'])
    y = data['y']

    # 4️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5️⃣ Handle imbalance
    ada = ADASYN(random_state=42)
    X_train_res, y_train_res = ada.fit_resample(X_train, y_train)

    # 6️⃣ Scale features
    X_train_scaled, X_test_scaled = transformer.scale_features(X_train_res, X_test)

    # 7️⃣ Model training
    trainer = ModelTrainer()
    results = trainer.train_models(X_train_scaled, y_train_res, X_test_scaled, y_test)

    # 8️⃣ Save best model (Random Forest assumed best)
    best_model = trainer.hyperparameter_tuning_rf(X_train_scaled, y_train_res)
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    run_training_pipeline()





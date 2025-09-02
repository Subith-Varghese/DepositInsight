from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
from src.logger import logger
import os
import joblib

import numpy


ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

    def train_models(self, x_train, y_train, x_test, y_test):
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            # Save the trained model
            file_name = f"{ARTIFACTS_DIR}/{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, file_name)
            logger.info(f"{name} saved at {file_name}")

            results[name] = {
                "model": model,
                "accuracy": accuracy_score(y_test, pred),
                "classification_report": classification_report(y_test, pred, output_dict=True)
            }
            logger.info(f"{name} trained with accuracy: {results[name]['accuracy']:.4f}")
        return results

    def hyperparameter_tuning_rf(self, x_train, y_train):
        param_grid_rf = {
            'n_estimators': [100, 150, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        rf = RandomForestClassifier()
        grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, verbose=2)
        grid_search_rf.fit(x_train, y_train)
        
        best_rf = grid_search_rf.best_estimator_
        # Save the tuned Random Forest
        file_name = f"{ARTIFACTS_DIR}/random_forest_best.pkl"
        joblib.dump(best_rf, file_name)
        logger.info(f"Tuned Random Forest saved at {file_name}")
        logger.info(f"Best RF params: {grid_search_rf.best_params_}")

        return best_rf

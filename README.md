## 1. Project Overview
This project predicts whether a client will subscribe to a term deposit based on historical marketing campaign data and client information.
It demonstrates a binary classification workflow with;

- preprocessing
- feature selection
- imbalance handling
- model training
- hyperparameter tuning
- prediction pipelines.

---

## 2. Project Structure

```
student_performance_predictor/
│
├── notebooks/                            # Jupyter notebooks
│   └── bank_marketing_analysis.ipynb
│
├── src/                                  # Source code
│   ├── __init__.py
│   ├── logger.py                         # Logging utility
│   ├── components/                       # Core modules
│   │   ├── __init__.py
│   │   ├── data_ingestion.py             # Load & separate dataset
│   │   ├── data_transformation.py        # Handle missing values, encode, scale,VIF-based feature selection
│   │   ├── feature_selection.py          # Chi-square 
│   │   ├── model_trainer.py              # Train & save models
│   │   ├── predictor.py                  # Load trained model for predictions
│   │
│   ├── pipelines/                        # Pipelines
│   │   ├── __init__.py
│   │   ├── training_pipeline.py          # End-to-end training
│   │   ├── predict_pipeline.py           # End-to-end prediction
│
├── artifacts/                            # Saved models, encoders, predictions
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── label_encoders.pkl
│   └── predictions.csv
│
├── data/                                 # Raw and new data
│   ├── bank-additional-full-1.csv
│   └── new_customers.csv
│
├── app.py                                # Flask web app (if deployed)
├── requirements.txt                      # Dependencies
└── README.md


```

## 3. Dataset Key Details
### 3.1 Target Variable → y
    - yes → Client subscribed to the term deposit✅.
    - no → Client did not subscribe❌.

### 3.2 Problem Type 
- Binary Classification

### 3.3 Imbalance Issue →
    - Majority: no 32,446
    - Minority: yes 2,731
    - Oversampling (ADASYN) was used in training to balance classes.

---
## 4. Features in the Dataset

The dataset has bank client info, economic indicators, and campaign details:
(a) Client-related Features

--- 

| Feature     | Description            | Example             |
| ----------- | ---------------------- | ------------------- |
| `age`       | Age of the client      | 35                  |
| `job`       | Job type               | admin, blue-collar  |
| `marital`   | Marital status         | married, single     |
| `education` | Education level        | secondary, tertiary |
| `default`   | Has credit in default? | yes / no            |
| `housing`   | Has housing loan?      | yes / no            |
| `loan`      | Has personal loan?     | yes / no            |

---

(c) Economic Indicators

| Feature          | Description                                     | Example |
| ---------------- | ----------------------------------------------- | ------- |
| `emp.var.rate`   | Employment variation rate (quarterly indicator) | 1.1     |
| `cons.price.idx` | Consumer price index                            | 93.994  |
| `cons.conf.idx`  | Consumer confidence index                       | -40     |
| `euribor3m`      | Euribor 3-month rate                            | 4.857   |
| `nr.employed`    | Number of employees                             | 5191    |

---

## 5. Main Aim of the Project

The primary aim is to build an ML model that predicts whether a client will subscribe to a term deposit based on their profile and previous campaign history.

---

## 6. Machine Learning Objective

- Problem Type: Binary Classification
- Input: Client information, campaign details, and economic indicators.
- Output: Predict yes or no (subscription status).
- Metric to Optimize:
    - Recall → Because we don’t want to miss customers who will subscribe.
    - F1-score → To balance precision & recall, since the dataset is imbalanced.

---

## 7. Project Workflow

Step-by-step workflow as implemented in the notebook:

1. Data Ingestion: Load dataset, separate rows with null target values for prediction.
2. Data Cleaning & Transformation:
    - Handle missing values
    - Group age by decade
    - Encode categorical variables (LabelEncoder saved to artifacts/)
    - Remove multicollinearity using VIF
    - Standard scaler : saved to artifacts/

3. Feature Selection:
    - Drop categorical columns via Chi-Square test

4. Train-Test Split & Imbalance Handling:
    - 80-20 split
    - ADASYN oversampling applied to training data

5. Model Training:
    - Logistic Regression, Decision Tree, Random Forest
    - Save all models to artifacts/

6. Hyperparameter Tuning:
    - Random Forest tuned using GridSearchCV
    - Best model saved as random_forest_best.pkl

7. Save Models: All models and label encoders saved to artifacts/

8. Prediction Pipeline:
    - Load new data
    - Apply same preprocessing
    - Make predictions
    - Save predictions in artifacts/predictions.csv

---

## 8. Setup & Installation

```
# Clone the repository
git clone https://github.com/Subith-Varghese/DepositInsight.git
cd student_performance_predictor

# Create virtual environment and activate
python -m venv venv
venv\Scripts\activate       

# Install dependencies
pip install -r requirements.txt

```
---
## 9. Running the Training Pipeline
```
python src/pipelines/training_pipeline.py
```

- This will train all models and save them in artifacts/
- Label encoders are also saved for prediction

---

## 10. Running the Prediction Pipeline
```
python src/pipelines/predict_pipeline.py

```
- Uses random_forest_best.pkl by default
- Loads unseen test data (data/new_customers.csv)
- Saves predictions in artifacts/predictions.csv

---
## 11. Model Evaluation
| Model               | Accuracy | Precision (yes) | Recall (yes) | F1-score (yes) | Notes                                         |
| ------------------- | -------- | --------------- | ------------ | -------------- | --------------------------------------------- |
| Decision Tree       | 0.96     | 0.79            | 0.82         | 0.80           | Good performance, interpretable               |
| Logistic Regression | 0.83     | 0.37            | 0.82         | 0.52           | Poor precision, struggles with minority class |
| Random Forest       | 0.96     | 0.85            | 0.82         | 0.84           | Best overall, handles imbalance well          |
| Random Forest Best  | 0.96     | 0.83            | 0.83         | 0.83           | Slightly tuned version, balanced metrics      |


## Summary

- Dataset Purpose: Customer behavior analysis for marketing.
- Project Goal: Predict if a customer subscribes to a term deposit.
- Business Benefit: Save cost, improve marketing efficiency.
- ML Task: Binary classification (yes or no).
- Best Model: Random Forest / XGBoost (depending on your final metrics).
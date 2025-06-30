# 🔍 MLflow-Tracked Random Forest Model with Hyperparameter Tuning

## 📌 Project Overview

This project demonstrates how to build and optimize a **Random Forest Regressor** with **Grid Search**, track experiments using **MLflow**, and log both model performance and parameters for reproducibility and experiment comparison.

---

## 🚀 Features

- 📊 Performs hyperparameter tuning with `GridSearchCV`
- 🧪 Tracks experiments using `mlflow.start_run()`
- 📁 Logs:
  - Best hyperparameters
  - Model performance (Mean Squared Error)
  - Model artifacts and signatures
- 💻 Works with **MLflow Tracking UI** (runs on localhost:5000)

---

## 🧱 Tech Stack

- Python
- Scikit-learn
- MLflow
- Pandas / NumPy (assumed for dataset prep)

---

## 🧠 Key Concepts

| Component        | Description                                         |
|------------------|-----------------------------------------------------|
| `train_test_split` | Splits the dataset into training and testing sets |
| `GridSearchCV`   | Searches over hyperparameter grid for best model   |
| `mlflow.log_param` | Logs hyperparameters to MLflow                    |
| `mlflow.log_metric` | Logs performance metric (MSE)                   |
| `mlflow.sklearn.log_model` | Saves the model for tracking & serving   |

---

## 📄 Code Highlights

### 🎯 Model Training & Tuning

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


Results:

<img width="940" alt="image" src="https://github.com/user-attachments/assets/77c8fe47-c612-4978-ac2d-17fe26a81a27" />



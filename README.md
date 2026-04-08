# 🏠 Boston Housing Price Prediction

A machine learning pipeline that predicts housing prices using the Boston Housing dataset. The project trains and compares two regression models — **Linear Regression** and **Ridge Regression** — and automatically exports the best-performing one for production use.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Output Files](#output-files)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Notes](#notes)
- [License](#license)

---

## Overview

This project demonstrates a standard end-to-end supervised ML pipeline:

1. Load and preprocess tabular data
2. Scale features using `StandardScaler`
3. Train two regression models in parallel
4. Evaluate both models using MSE and R²
5. Select and serialize the best model
6. Generate diagnostic visualizations

---

## Project Structure

```
boston-housing-prediction/
│
├── boston.csv                  # Dataset (place in root directory)
├── main.py                     # Main pipeline script
│
├── scaler.pkl                  # Saved StandardScaler (generated)
├── best_model.pkl              # Best trained model (generated)
│
├── heatmap.png                 # Feature correlation heatmap (generated)
└── prediction_vs_actual.png    # Actual vs predicted scatter plot (generated)
```

---

## Prerequisites

- Python 3.8+
- The dataset saved as `boston.csv` in the project root

### Required Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, scaling, evaluation |
| `joblib` | Model serialization |
| `seaborn` | Statistical visualizations |
| `matplotlib` | Plot generation |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/boston-housing-prediction.git
cd boston-housing-prediction

# Install dependencies
pip install pandas numpy scikit-learn joblib seaborn matplotlib
```

> **Note:** The Boston Housing dataset has been removed from scikit-learn (v1.2+) due to ethical concerns. You will need to supply your own `boston.csv` with a `medv` target column, or adapt the script to use `fetch_california_housing()` instead.

---

## Usage

```bash
python main.py
```

The script will:
- Train both models on 80% of the data
- Evaluate both on the remaining 20%
- Print performance metrics to the console
- Save the best model and visualizations to the project directory

### Sample Console Output

```
--- Model Performance ---
Linear Regression  -> MSE: 24.291, R2: 0.669
Ridge Regression   -> MSE: 23.987, R2: 0.673

Result: Ridge Regression selected as the best model.
Best model saved as 'best_model.pkl'

Visualizations saved successfully.
```

---

## Pipeline Stages

### 1. Load Dataset
Reads `boston.csv` into a pandas DataFrame. Exits gracefully if the file is not found.

### 2. Preprocessing
- Separates features (`X`) from the target column `medv`
- Splits data into train/test sets (`test_size=0.2`, `random_state=42`)
- Applies `StandardScaler` to normalize features
- Saves the fitted scaler as `scaler.pkl` for use during inference

### 3. Model Training
Two models are trained on the scaled training data:
- **Linear Regression** — baseline, no regularization
- **Ridge Regression** — L₂ regularization with `alpha=1.0` to reduce overfitting

### 4. Evaluation
Both models are evaluated on the held-out test set using:
- **MSE** (Mean Squared Error) — lower is better
- **R²** (Coefficient of Determination) — higher is better

### 5. Model Selection & Export
The model with the higher R² score is serialized to `best_model.pkl` using `joblib`.

### 6. Visualization
Two plots are saved automatically:
- **Correlation Heatmap** — shows pairwise feature relationships
- **Prediction vs Actual** — scatter plot of model accuracy on the test set

---

## Output Files

| File | Description |
|---|---|
| `scaler.pkl` | Fitted `StandardScaler` — must be applied to any new input data before inference |
| `best_model.pkl` | Best-performing regression model (Linear or Ridge) |
| `heatmap.png` | Correlation matrix of all features |
| `prediction_vs_actual.png` | Scatter plot comparing predicted vs true housing prices |

---

## Model Performance

| Metric | Linear Regression | Ridge Regression |
|---|---|---|
| MSE | ~24.3 | ~24.0 |
| R² | ~0.669 | ~0.673 |

> Results may vary slightly depending on the dataset version used.

---

## Visualizations

**Feature Correlation Heatmap**
Shows how features like room count (`rm`) or crime rate (`crim`) correlate with the target variable `medv`.

**Prediction vs Actual Scatter Plot**
Points close to the red dashed line indicate more accurate predictions. Spread from the line reveals model error.

---

## Notes

- **Inference:** Always load `scaler.pkl` and apply `.transform()` to new data before passing it to `best_model.pkl`.
- **Reproducibility:** The train/test split uses `random_state=42` for consistent results.
- **Extending the pipeline:** Consider adding cross-validation, hyperparameter tuning (e.g., `GridSearchCV`), or additional models like `Lasso` or `RandomForestRegressor` for comparison.

---

## License

This project is licensed under the [MIT License](LICENSE).

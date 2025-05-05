# Bank Deposit Prediction with Machine Learning

**Developed as part of a university project at the University of Salerno** for the Machine Learning course.
This project aims to predict whether a client will subscribe to a term deposit based on historical marketing campaign data from a bank. The task is a **binary classification problem**.

## Project Structure
- ├── bank.csv
- ├── bank.ipynb 
- ├── requirements.txt
- ├── README.md 

# Dataset
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

The dataset is available at the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## Objective

To build and evaluate machine learning models capable of predicting customer subscription to a term deposit (`y`: yes/no), with attention to **class imbalance**, **feature selection**, and **model interpretability**.

---

## Technologies and Libraries

- Python 3.10
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Jupyter Notebook

---
## ⚙️ Virtual Environment Setup (optional but recommended)

To isolate dependencies and ensure reproducibility:

```bash
# Create a virtual environment
python -m venv EVbank

# Activate the environment
# On Windows:
.\EVbank\Scripts\activate
# On Mac/Linux:
source EVbank/bin/activate

# Install project dependencies
pip install -r requirements.txt
```
---

## Data Preprocessing

- Replaced `unknown` values with NaN and imputed using the mode.
- Dropped the `poutcome` feature due to excessive missing values (over 88%).
- Handled missing values in `contact` by introducing a new category: `no_contact`.
- Converted the target variable `y` into binary (1 = yes, 0 = no).
- One-Hot Encoding for categorical variables.
- Feature selection with `SelectKBest` (f_classif) to select top 10 features.

---

## Handling Class Imbalance

- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the training set.

---

## Models Used

| Model               | Description |
|--------------------|-------------|
| Random Forest       | Robust ensemble method using bagging |
| Gradient Boosting   | Sequential ensemble using boosting and gradient descent |
| Logistic Regression | Interpretable baseline model |
| Naive Bayes         | Probabilistic model with strong independence assumptions |

### Why these models?

- To cover a mix of **interpretable**, **ensemble**, and **probabilistic** approaches.
- Gradient Boosting and Random Forest for **accuracy and performance**.
- Logistic Regression and Naive Bayes for **speed and interpretability**.

---

## Evaluation Metrics

- **Accuracy**
- **F1-Score**
- **ROC AUC**
- **Confusion Matrix**
- **Classification Report**

---

## Results

The Gradient Boosting model achieved the highest ROC AUC score, showing the best ability to distinguish between the two classes, although all models showed similar accuracy.

| Model              | Accuracy | F1 Score | ROC AUC |
|-------------------|----------|----------|---------|
| Gradient Boosting | ~0.80    | ~0.48    | **Best** |
| Random Forest     | ~0.81    | ~0.46    | Good    |
| Logistic Regression | ~0.80 | ~0.46    | Moderate |
| Naive Bayes       | ~0.65    | ~0.32    | Low     |

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Use of advanced models like **XGBoost** or **LightGBM**.
- Model explanation with **SHAP** values.
- Deploy a web app or interactive dashboard for business use.


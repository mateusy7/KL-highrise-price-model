# KL Highrise Price Predictor

A ML regression model that gives a fair price of residential highrise units in Kuala Lumpur
based on historical JPPH transaction data.

🔗 **[Live Demo Site](https://kl-highrise-price.streamlit.app/)**

---

## Project Overview

* **Target:** Transaction Price (RM)
* **Training features:** Property type, Mukim, Tenure (Leasthold, Freehold), Unit Level, Land parcel area (sqft), Mukim median income, Scheme name/Area median price
* **Data source:** [Property sales data](https://napic.jpph.gov.my/en/open-sales-data), [Income data](https://data.gov.my/data-catalogue/hh_income_parlimen)
* **Data timeframe:** 2021 to 2025 (25,978 entries)
* **ML libraries:** Pandas, Numpy, Scikit-Learn, Matplotlib
* **Deployment:** FastAPI on AWS ECS and AWS App Runner (Backend), Streamlit Cloud (Frontend)

---

## Methodology

1. **Data Engineering:** Cleaned historical transaction data from JPPH, mapped income and population data from DOSM to each of 8 'mukims (district)' in KL. 
2. **Preprocessing** One-hot encoding for 'property_type', 'mukim' and 'tenure' features, leave-one-out target encoding of 1000+ highrise buildings/areas to the median price of top 30 most frequent sale regions within each 'mukim', transaction price capped at the `99th percentile (RM 4,200,000)` to limit effect of outliers during training. For reference: `median or 50th percentile (RM 470,000)` and `99.9th percentile (RM9,200,000)`.
3. **Model Training:** Trained 3 models: polynomial regression, decision tree, and random forest model with custom full training Scikit-learn pipelines, hyperparameter tuning with grid search CV on 5-folds training data for RMSE performance.
4. **Model Evaluation:** Random forest model had lowest test `RMSE of RM 194,000`. However, RMSE is highly skewed by large outliers data, `MAPE (mean absolute percentage error) at 15.9%` shows a better individual performance measure, which is an average of a 15.9% price deviation from the true price.

---

## Evaluation Metrics (Test set)
| Metric: Value |  
| RMSE (overall): `RM 194,000` | RMSE (80th percentile, RM 1,000,000 and below): `RM 113,000` |  
| MAPE (overall): `     15.9%` | MAPE (80th percentile, RM 1,000,000 and below): `     16.7%` |

**Model Type**: Random Forest  
**Validation Strategy**: 80/20 Train-Test Split with 5-fold CV on training set

> **Note:**  
Property prices are highly right-skewed as shown in 04_feature_engineering.ipynb.  
Look at the end of 05_training.ipynb for more performance metrics.

---

## Repository Main Structure
```text
.
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for EDA and model development
└── src/                    # Main source directory
    ├── app/                # FastAPI backend
    ├── data/               # Data ingestion scripts (cleaning.py, load_data.py)
    ├── model/              # Model scripts (train.py, evaluate.py, inference.py)
    ├── preprocess/         # Data preprocessing scipts (feature_eng.py, preprocessing.py)
    ├── run_pipeline.py     # Script to execute the full pipeline
    └── streamlit_frontend.py # Streamlit UI
├── tests/                  # Tests for cleaning pipeline, inference logic
```

---

## Installation & Setup

Follow these steps to run the environment locally or redeploy the pipeline.

1. **Clone the repository**
   ```bash
   git clone https://github.com/mateusy7/KL-highrise-price-model.git
   cd KL-highrise-price-model
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run training pipeline to get trained model**
   ```bash
   python src/run_pipeline.py
   ```
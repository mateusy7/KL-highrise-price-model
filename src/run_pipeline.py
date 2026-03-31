import sys
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_PATH = str(BASE_DIR/"src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from data.load_data import load_data
from data.cleaning import clean_data
from preprocess.feature_eng import add_income_data, cap_outliers
from preprocess.preprocessing import split_data_stratify, prepare_train_set, prepare_test_set
from model.train import train_model
from model.evaluate import evaluate_model

DATA_PATH = BASE_DIR/"data"

def run_full_pipeline():

    # data ingestion
    df = load_data(DATA_PATH/"kuala_lumpur_data.csv")
    df_income = load_data(DATA_PATH/"mukim_income.csv")

    # clean data
    df_highrise = clean_data(df)

    # feature engineering
    df_highrise = add_income_data(df_highrise, df_income)
    df_highrise = cap_outliers(df_highrise)

    train_set, test_set = split_data_stratify(df_highrise)

    X_train, y_train, area_price_mapping = prepare_train_set(train_set)
    X_test, y_test = prepare_test_set(test_set, area_price_mapping)

    # preprocess data and train model
    model_trained = train_model(X_train, y_train)
    train_error, test_error = evaluate_model(model_trained,X_train, y_train, X_test, y_test)

    joblib.dump(area_price_mapping, BASE_DIR/"saved_models"/"area_price_mapping.pkl")

if __name__ == "__main__":
    run_full_pipeline()
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from preprocess.feature_eng import fit_area_median_price_looe, transform_test_set

def split_data_stratify(df: pd.DataFrame, stratify_col: str='mukim'):
    train_set, test_set = train_test_split(df,
                                           train_size=0.8,
                                           stratify=df[stratify_col],
                                           random_state=1)
    
    return train_set, test_set

def prepare_train_set(train_set):
    train_set = train_set.copy()
    train_set_looe, area_price_mapping = fit_area_median_price_looe(train_set)

    X_train = train_set_looe.drop(columns=['transaction_price', 'year', 'scheme_name_area'])
    y_train = train_set_looe['transaction_price']

    return X_train, y_train, area_price_mapping

def prepare_test_set(test_set, area_price_mapping):
    test_set = test_set.copy()
    test_set_prepped = transform_test_set(test_set, area_price_mapping)

    X_test = test_set_prepped.drop(columns=['transaction_price', 'year', 'scheme_name_area'])
    y_test = test_set_prepped['transaction_price']

    return X_test, y_test

num_cols = ['land_parcel_area', 'unit_level', 'income_median_area', 'area_median_price']
cat_cols = ['property_type', 'mukim', 'tenure']

def preprocess_linear(d=1):
    log_transform = ColumnTransformer([
        ("log_transform",
         FunctionTransformer(np.log, feature_names_out="one-to-one"),
         ['land_parcel_area', 'area_median_price'])
    ], remainder='passthrough')

    num_pipeline = Pipeline([
        ("log", log_transform),
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("scaler", StandardScaler())
    ])

    preprocess = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_cols),
        ("cat_encode", OneHotEncoder(), cat_cols)
    ])

    return preprocess

def preprocess_tree():
    log_pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log, feature_names_out="one-to-one")),
        ("scaler", StandardScaler())
    ])

    preprocess = ColumnTransformer([
        ("log", log_pipeline, ['land_parcel_area', 'area_median_price']),
        ("scaler", StandardScaler(), ['unit_level', 'income_median_area']),
        ("cat_encode", OneHotEncoder(), cat_cols)
    ])
    
    return preprocess
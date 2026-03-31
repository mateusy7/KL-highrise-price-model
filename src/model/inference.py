from pathlib import Path
import sys
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

ROOT_PATH = BASE_DIR
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

MODEL_PATH = BASE_DIR / "saved_models" / "forest_v1.pkl"
MAPPING_PATH = BASE_DIR / "saved_models" / "area_price_mapping.pkl"

from src.preprocess.feature_eng import transform_test_set

INCOME_MEDIAN_AREA = {
    "Kuala Lumpur Town Centre": 10655.95,
    "Mukim Ampang": 10573.02,
    "Mukim Batu": 11745.30,
    "Mukim Cheras": 9720.34,
    "Mukim Kuala Lumpur": 11817.21,
    "Mukim Petaling": 10346.59,
    "Mukim Setapak": 10463.18,
    "Mukim Ulu Kelang": 11180.00
}

EXPECTED_TRANSFORMED_COLUMNS = [
        'property_type', 'mukim', 'tenure',
        'land_parcel_area', 'unit_level', 'income_median_area',
        'area_median_price'
        ]

class ModelInference:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        # Load using the absolute path
        self.model = joblib.load(MODEL_PATH)

        if not MAPPING_PATH.exists():
            raise FileNotFoundError(f"Mapping not found at {MAPPING_PATH}")
        
        self.area_price_mapping = joblib.load(MAPPING_PATH)

    def predict(self, input_data: dict):
        """
        Returns highrise unit price prediction from prepared user query.

        Args:
            input_data (dict) : A dictionary containing:
            - property_type (str): Type of high-rise (e.g., 'Condo', 'Flat').
            - mukim (str): District in Kuala Lumpur.
            - scheme_name_area (str): Specific neighborhood or building name.
            - tenure (str): 'Freehold' or 'Leasehold'.
            - land_parcel_area (float): Size in square feet.
            - unit_level (int): Floor number of the unit.

        Returns:
            prediction (float) : highrise unit price prediction
        """

        query = self.prepare_query(input_data)

        prediction = self.model.predict(query[EXPECTED_TRANSFORMED_COLUMNS])
        
        return prediction
    
    def prepare_query(self, raw_user_query: dict):
        """
        Transforms the input query data from the user into a prepared query
        for model inferencing.

        Args:
            raw_user_query (dict) : A dictionary containing:
            - property_type (str): Type of high-rise (e.g., 'Condo', 'Flat').
            - mukim (str): District in Kuala Lumpur.
            - scheme_name_area (str): Specific neighborhood or building name.
            - tenure (str): 'Freehold' or 'Leasehold'.
            - land_parcel_area (float): Size in square feet.
            - unit_level (int): Floor number of the unit.

        Returns:
            prepared_query (DataFrame) :
            - property_type (str): Type of high-rise (e.g. 'Condo', 'Flat').
            - mukim (str): District in Kuala Lumpur.
            - tenure (str): 'Freehold' or 'Leasehold'.
            - land_parcel_area (float): Size in square feet.
            - unit_level (int): Floor number of the unit.
            - income_median_area (float): median income in the query's 'mukim' for 2025
            - area_median_price (float) : median past transaction prices of highrise units in neighborhood area
        """
        query = pd.DataFrame([raw_user_query])

        query['income_median_area'] = query['mukim'].map(INCOME_MEDIAN_AREA)
        query = transform_test_set(query, self.area_price_mapping)
        query_transformed = query.drop(columns=['scheme_name_area'])

        query_prepared = query_transformed[EXPECTED_TRANSFORMED_COLUMNS]
        
        return query_prepared
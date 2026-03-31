import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ROOT_PATH = str(BASE_DIR)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

DATA_PATH = BASE_DIR/"data"

from src.data.load_data import load_data
from src.data.cleaning import clean_data

def main():
    df = load_data(DATA_PATH/"kuala_lumpur_data.csv")
    print(f"Data loaded. \nSample:\n\n{df.head(2)}\n")

    print(f"Cleaning data...\n")

    df_highrise = clean_data(df)
    print(f"Property Types: {df_highrise['property_type'].unique().tolist()}")
    print(f"Columns: {df_highrise.columns.to_list()}")
    print(f"Data Types: {[str(t) for t in df_highrise.dtypes]}")
    print(f"NaN values: {df_highrise.isna().sum().to_list()}\n")

    print("Data Cleaning Completed.\n")

if __name__ == "__main__":
    main()
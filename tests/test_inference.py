import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ROOT_PATH = str(BASE_DIR)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

DATA_PATH = BASE_DIR / "data"

from src.model.inference import ModelInference
import pandas as pd

def main():
    model = ModelInference()

    df = pd.read_csv(DATA_PATH/"train_set.csv")
    sample = df.loc[4].drop(index=['transaction_price', 'year', 'income_median_area', 'area_median_price']).to_dict()

    print(f"\nSample before:\n {sample})\n")
    print(f"Sample after:\n {model.prepare_query(sample)}\n")
    print(f"Model Prediction: {model.predict(sample)[0].astype(int)}, True Price: {df.loc[4, 'transaction_price']}\n")

if __name__ == "__main__":
    main()
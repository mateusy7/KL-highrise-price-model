import pandas as pd
import re

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.replace(r'[ ,/]', '_', regex=True)

    df.rename(columns={'month__year_of_transaction_date': 'month_year'}, inplace=True)
    df['month_year'] = pd.to_datetime(df['month_year'], format='%B %Y')

    df['year'] = df['month_year'].dt.year
    df.drop(columns='month_year', inplace=True)
    df.drop('district', axis=1, inplace=True)

    mask = df.iloc[:, 8] == '-'
    df_highrise = df[mask].copy()

    df_highrise.drop(['road_name', 'unit', 'main_floor_area'], axis=1, inplace=True)

    df_highrise['unit_level'] = df_highrise['unit_level'].apply(clean_floor)

    sqm_to_sqft = 10.7639104
    df_highrise['land_parcel_area'] = df_highrise['land_parcel_area'] * sqm_to_sqft

    df_highrise['transaction_price'] = df_highrise['transaction_price'].astype(float)

    return df_highrise

def clean_floor(level: pd.Series):
    original_input = str(level).strip().upper()
    
    mapping = {
        '3A': '4', '13A': '14', '23A': '24', 'G': '0', 'B': '-1',
        'MZ': '0', 'D':'0', 'UG': '0', 'P': '0', 'LG': '0'
    }
    
    current_val = mapping.get(original_input, original_input)
    
    match = re.search(r'\d+', current_val)
    
    return int(match.group()) if match else None
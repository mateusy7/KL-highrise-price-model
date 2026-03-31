import pandas as pd

expected_cols = ['property_type', 'mukim', 'scheme_name_area', 'tenure', 'land_parcel_area',
                 'unit_level', 'transaction_price', 'year', 'income_median_area']

def add_income_data(df_highrise: pd.DataFrame, df_income: pd.DataFrame) -> pd.DataFrame:

    df = df_highrise.merge(df_income, on=['year', 'mukim'], how='left')
    df = df.rename(columns={"income_median_weighted":"income_median_area"})

    return df

def cap_outliers(df_highrise: pd.DataFrame,
                 area_cap: float=4100,
                 area_cap_amount: float=4200,
                 price_cap: float=4.3e6,
                 price_cap_amount: float=4.4e6) -> pd.DataFrame:\
    
    df = df_highrise.copy()
    assert set(expected_cols).issubset(df.columns), f"Missing columns: {set(expected_cols) - set(df.columns)}"
    
    df.loc[df['land_parcel_area'] > area_cap, 'land_parcel_area'] = area_cap_amount
    df.loc[df['transaction_price'] > price_cap, 'transaction_price'] = price_cap_amount

    return df

def fit_area_median_price_looe(
    train_set: pd.DataFrame,
    price_col: str = 'transaction_price',
    group_cols: list = ['mukim', 'scheme_name_area'],
    min_count: int = 10,
    top_n: int = 30
) -> tuple:
    """
    Leave one out target encoding of the area name in 
    each mukim with the median price of the area.
    """
    df = train_set.copy()

    # 1. Logic for identifying 'Outside' areas
    # Exact notebook logic: value_counts -> head(30) -> filter < 10 or not in top_30
    area_count = df.groupby(group_cols[0])[group_cols[1]].value_counts()
    top_30_idx = area_count.groupby(level=0).head(top_n).index
    
    outside = area_count[(area_count < min_count) | (~area_count.index.isin(top_30_idx))]
    outside_areas_list = outside.index.tolist()

    # Create the boolean mask
    is_outside_row = df.set_index(group_cols).index.isin(outside_areas_list)

    # 2. Define LOO helpers (Internal to keep function clean)
    def loo_median(x):
        return pd.Series([x.drop(i).median() for i in x.index], index=x.index)

    def compute_mukim_others(group):
        prices = group[price_col]
        return pd.Series([prices.drop(i).median() for i in prices.index], index=prices.index)

    # 3. Apply LOOE to standard groups
    # Note: reset_index(level=[0, 1], drop=True) works here assuming DF index is standard
    df['area_median_price'] = (
        df.groupby(group_cols)[price_col]
        .apply(loo_median)
        .reset_index(level=[0, 1], drop=True)
    )

    # 4. Handle 'Others' grouping logic
    if is_outside_row.any():
        others_looe_series = (
            df[is_outside_row]
            .groupby(group_cols[0], group_keys=False)
            .apply(compute_mukim_others)
        )
        df.loc[is_outside_row, 'area_median_price'] = others_looe_series
        df.loc[is_outside_row, 'scheme_name_area'] = 'Others'

    # 5. Build Mappings for Test Set (to match your notebook's logic for UI)
    area_mapping = df.groupby(group_cols)['area_median_price'].median().to_dict()
    
    others_value_by_mukim = (
        df[df[group_cols[1]] == 'Others']
        .groupby(group_cols[0])['area_median_price']
        .median()
        .to_dict()
    )
    
    global_median = df['area_median_price'].median()

    mapping_bundle = {
        'main': area_mapping,
        'mukim_others': others_value_by_mukim,
        'global': global_median
    }

    return df, mapping_bundle

def transform_test_set(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Applies the learned mappings to the test set or UI input.
    """
    df = df.copy()
    
    def map_row(row):
        key = (row['mukim'], row['scheme_name_area'])
        # Try specific area
        if key in mapping['main']:
            return mapping['main'][key]
        # Try mukim-level fallback
        return mapping['mukim_others'].get(row['mukim'], mapping['global'])

    df['area_median_price'] = df.apply(map_row, axis=1)
    return df
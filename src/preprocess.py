import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib, os

# Columns that MUST be dropped — data leakage
# NOTE: Late_delivery_risk is NOT here — it's the target,
# handled separately inside encode_and_split()
LEAKAGE_COLS = [
    'Days for shipping (real)',
    'Delivery Status',
]

# Columns to drop — IDs, high-cardinality text, duplicates
DROP_COLS = [
    'Order Id', 'Order Item Id', 'Order Customer Id',
    'Customer Id', 'Customer Fname', 'Customer Lname',
    'Customer Email', 'Customer Password', 'Customer Street',
    'Customer Zipcode', 'Product Name', 'Product Description',
    'Product Image', 'order date (DateOrders)',
    'shipping date (DateOrders)', 'derived_risk','order_date',
]

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding='unicode_escape')

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Parse dates
    df['order_date']    = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
    df['order_month']   = df['order_date'].dt.month
    df['order_year']    = df['order_date'].dt.year
    df['order_quarter'] = df['order_date'].dt.quarter

    # 2. Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['order_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['order_month'] / 12)

    # 3. Drop leakage and ID columns
    # Late_delivery_risk is deliberately NOT in this list
    cols_to_drop = [c for c in LEAKAGE_COLS + DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 4. Drop remaining high-cardinality string columns
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() > 200:
            df.drop(columns=[col], inplace=True)

    return df

def encode_and_split(df: pd.DataFrame, target_col: str, test_year: int = 2019):
    df = df.copy()

    # Extract target FIRST before touching the dataframe
    y = df[target_col].values
    df.drop(columns=[target_col], inplace=True, errors='ignore')

    # Label-encode categoricals
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Fill nulls
    df.fillna(df.median(numeric_only=True), inplace=True)

    feature_names = df.columns.tolist()

    # Temporal split
    train_mask = df['order_year'] < test_year
    test_mask  = df['order_year'] >= test_year

    if test_mask.sum() == 0:
        available_years = sorted(df['order_year'].dropna().astype(int).unique())
        if not available_years:
            raise ValueError('No valid order_year values found for temporal split.')

        fallback_year = available_years[-2] if len(available_years) > 1 else available_years[-1]
        print(
            f'No rows found for test_year >= {test_year}; '
            f'using test_year={fallback_year} instead.'
        )
        test_year = fallback_year
        train_mask = df['order_year'] < test_year
        test_mask  = df['order_year'] >= test_year

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f'Invalid temporal split for test_year={test_year}: '
            f'train rows={train_mask.sum()}, test rows={test_mask.sum()}.'
        )

    X_train = df[train_mask].values
    X_test  = df[test_mask].values
    y_train = y[train_mask]
    y_test  = y[test_mask]

    return X_train, X_test, y_train, y_test, feature_names, encoders

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import engineer_features, load_raw


def main():
    data_dir = PROJECT_ROOT / 'data'
    models_dir = PROJECT_ROOT / 'models'

    raw = load_raw(data_dir / 'DataCoSupplyChainDataset.csv')
    engineered = engineer_features(raw)

    model = joblib.load(models_dir / 'xgb_model.pkl')
    high_risk_df = pd.read_csv(data_dir / 'high_risk_orders.csv')
    cluster_profiles = pd.read_csv(data_dir / 'cluster_profiles.csv')

    encoded_full = engineered.drop(columns=['Late_delivery_risk'], errors='ignore')
    for col in encoded_full.select_dtypes(include='object').columns:
        encoded_full[col] = LabelEncoder().fit_transform(encoded_full[col].astype(str))
    encoded_full.fillna(encoded_full.median(numeric_only=True), inplace=True)

    y_prob = model.predict_proba(encoded_full.values)[:, 1]
    export_raw = raw.copy()
    export_raw['predicted_risk'] = y_prob
    export_raw['cluster'] = -1
    export_raw['order_month'] = engineered['order_month'].values
    export_raw['order_year'] = engineered['order_year'].values

    if 'Order Id' in high_risk_df.columns and 'Order Id' in export_raw.columns:
        cluster_map = high_risk_df.set_index('Order Id')['cluster']
        export_raw['cluster'] = (
            export_raw['Order Id'].map(cluster_map).fillna(-1).astype(int)
        )
    else:
        high_risk_mask = y_prob >= 0.6
        if high_risk_mask.sum() != len(high_risk_df):
            raise ValueError(
                'High-risk row count mismatch: '
                f'raw/model={high_risk_mask.sum()}, clustering CSV={len(high_risk_df)}'
            )
        export_raw.loc[high_risk_mask, 'cluster'] = high_risk_df['cluster'].values

    export_cols = [
        'Order Id', 'Order Region', 'Market', 'Shipping Mode',
        'Department Name', 'Category Name', 'Customer Segment',
        'Late_delivery_risk', 'predicted_risk', 'cluster',
        'Order Profit Per Order', 'order_month', 'order_year',
    ]

    export_df = export_raw[export_cols].copy()
    export_df.to_csv(data_dir / 'powerbi_export.csv', index=False)
    cluster_profiles.to_csv(data_dir / 'cluster_profiles.csv', index=False)
    print('Exported for Power BI.')
    print(f'  {data_dir / "powerbi_export.csv"} ({len(export_df):,} rows)')
    print(f'  {data_dir / "cluster_profiles.csv"} ({len(cluster_profiles):,} rows)')
    print(f'Late delivery rate: {export_df["Late_delivery_risk"].mean():.1%}')
    print(f'High risk orders: {(export_df["predicted_risk"] >= 0.6).sum():,}')


if __name__ == '__main__':
    main()

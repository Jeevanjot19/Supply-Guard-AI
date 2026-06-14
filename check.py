import pandas as pd
df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='unicode_escape')
df['derived_risk'] = (df['Days for shipping (real)'] > df['Days for shipment (scheduled)']).astype(int)
print(f"Target match rate: {(df['derived_risk'] == df['Late_delivery_risk']).mean():.4f}")
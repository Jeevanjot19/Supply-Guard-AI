import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# def find_optimal_k(X_scaled, k_range=range(2, 11)):
#     inertias, sils = [], []
#     for k in k_range:
#         km = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = km.fit_predict(X_scaled)
#         inertias.append(km.inertia_)
#         sils.append(silhouette_score(X_scaled, labels))

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#     ax1.plot(list(k_range), inertias, 'bo-')
#     ax1.set_title('Elbow Method'); ax1.set_xlabel('K')
#     ax2.plot(list(k_range), sils, 'ro-')
#     ax2.set_title('Silhouette Score'); ax2.set_xlabel('K')
#     plt.tight_layout()
#     from pathlib import Path
#     plt.savefig(Path(__file__).resolve().parent.parent / 'data' / 'cluster_k_selection.png', dpi=150)
#     plt.show()

#     best_k = list(k_range)[np.argmax(sils)]
#     print(f'Best K by silhouette: {best_k}')
#     return best_k

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, 'bo-')
    ax1.set_title('Elbow Method'); ax1.set_xlabel('K')
    ax2.plot(list(k_range), sils, 'ro-')
    ax2.set_title('Silhouette Score'); ax2.set_xlabel('K')
    plt.tight_layout()

    # Save relative to project root — works regardless of where notebook runs from
    save_path = Path(__file__).resolve().parent.parent / 'data' / 'cluster_k_selection.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()

    best_k = list(k_range)[np.argmax(sils)]
    print(f'Best K by silhouette: {best_k}')
    return best_k


def cluster_on_shap(df_full: pd.DataFrame,
                    shap_values: np.ndarray,
                    feat_names: list,
                    y_prob: np.ndarray,
                    risk_threshold: float = 0.6,
                    k: int = None):
    """
    Cluster high-risk orders on their SHAP value vectors.
    Groups orders by WHY they are risky, not just what their features are.

    Args:
        df_full:     original engineered df with string columns (for profile labels)
        shap_values: numpy array shape (n_orders, n_features) — from TreeExplainer
        feat_names:  list of feature names matching shap_values columns
        y_prob:      predicted risk probabilities aligned with df_full rows
        risk_threshold: filter threshold for high-risk orders
        k:           number of clusters (None = auto-select via silhouette)
    """
    df_full = df_full.copy()
    df_full['predicted_risk'] = y_prob

    # Filter to high-risk orders
    high_risk_mask = df_full['predicted_risk'] >= risk_threshold
    high_risk = df_full[high_risk_mask].copy()
    shap_hr   = shap_values[high_risk_mask]   # SHAP rows for high-risk only

    print(f'High-risk orders: {len(high_risk)} ({high_risk_mask.mean()*100:.1f}% of total)')

    # Build SHAP DataFrame for clustering
    shap_df = pd.DataFrame(shap_hr, columns=feat_names, index=high_risk.index)

    # Scale SHAP values — different features have different SHAP magnitudes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(shap_df)

    # Auto-select K if not provided
    if k is None:
        k = find_optimal_k(X_scaled)

    # Fit K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    high_risk['cluster'] = km.fit_predict(X_scaled)

    # Build cluster profiles
    profiles = []
    for cluster_id in sorted(high_risk['cluster'].unique()):
        subset      = high_risk[high_risk['cluster'] == cluster_id]
        subset_shap = shap_df.loc[subset.index]

        # Dominant SHAP driver = feature with highest mean |SHAP| in this cluster
        mean_abs_shap = subset_shap.abs().mean()
        dominant_shap_feature = mean_abs_shap.idxmax()
        dominant_shap_value   = mean_abs_shap.max().round(4)

        profile = {
            'cluster_id':             cluster_id,
            'order_count':            len(subset),
            'avg_risk_score':         round(float(subset['predicted_risk'].mean()), 3),
            'dominant_shap_feature':  dominant_shap_feature,   # WHY they are risky
            'dominant_shap_value':    dominant_shap_value,
            'dominant_shipping_mode': subset['Shipping Mode'].mode()[0] if 'Shipping Mode' in subset.columns else 'N/A',
            'dominant_region':        subset['Order Region'].mode()[0]   if 'Order Region'  in subset.columns else 'N/A',
            'dominant_department':    subset['Department Name'].mode()[0] if 'Department Name' in subset.columns else 'N/A',
            'dominant_month':         int(subset['order_month'].mode()[0]) if 'order_month' in subset.columns else 0,
            'profit_at_risk':         round(float(subset['Order Profit Per Order'].fillna(0).sum()), 2),
        }
        profiles.append(profile)
        print(f'  Cluster {cluster_id}: {len(subset)} orders | '
              f'Risk driver: {dominant_shap_feature} (SHAP={dominant_shap_value}) | '
              f'Region: {profile["dominant_region"]} | Mode: {profile["dominant_shipping_mode"]}')

    # Save artifacts
    

    project_root = Path(__file__).resolve().parent.parent
    models_dir   = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(km,     models_dir / 'kmeans_model.pkl')
    joblib.dump(scaler, models_dir / 'cluster_scaler.pkl')
    joblib.dump(shap_df, models_dir / 'shap_hr_df.pkl')
    return high_risk, pd.DataFrame(profiles)

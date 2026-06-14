from typing import TypedDict, Optional

class InvestigationState(TypedDict):
    # Input
    cluster_profile: dict          # From cluster_profiles.csv
    full_data_path: str            # Path to high_risk_orders.csv

    # Findings (populated by nodes)
    regional_finding: str
    mode_finding: str
    seasonal_finding: str
    profit_at_risk: float

    # Router output
    primary_cause: str             # 'regional' | 'mode' | 'seasonal' | 'combined'
    severity: str                  # 'high' | 'medium' | 'low'

    # Final output
    recommendation_memo: str

    # Trace (for Streamlit display)
    trace: list                    # list of strings — one per completed node

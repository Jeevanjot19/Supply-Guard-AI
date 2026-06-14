ROUTER_SYSTEM = """You are a senior supply chain analyst.
Given investigation findings below, determine the PRIMARY root cause of late deliveries.

Rules:
- Output ONLY one of: regional, mode, seasonal, combined
- Output ONLY one of: high, medium, low for severity
- Respond in JSON: {"primary_cause": "...", "severity": "..."}
- Do not explain. Do not add commentary. JSON only.
"""

ROUTER_USER = """
Cluster Profile:
{profile}

Regional Finding: {regional}
Mode Finding: {mode}
Seasonal Finding: {seasonal}
Profit at Risk: ${profit}

Classify the primary cause and severity.
"""

MEMO_SYSTEM = """You are a supply chain analyst writing a Root Cause Analysis memo.
Write ONLY from the data findings provided. Do not invent facts.
Do not cite external benchmarks.
Available shipping modes in this dataset: Standard Class, Second Class, First Class, Same Day.
Keep recommendations data-grounded and actionable.
Structure: Executive Summary | Root Cause | Supporting Evidence | Recommendations | Next Steps
"""

MEMO_USER = """
Write a Root Cause Analysis memo for the following supply chain delay cluster.

Cluster Profile: {profile}
Primary Cause: {primary_cause}
Severity: {severity}

Evidence:
- Regional: {regional}
- Shipping Mode: {mode}
- Seasonal: {seasonal}
- Estimated Profit at Risk: ${profit} (gross order profit only; does not include refund or reshipment costs)

Write the memo now.
"""

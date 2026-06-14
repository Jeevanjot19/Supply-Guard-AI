
# SupplyGuard AI
### Autonomous Supply Chain Risk Investigation Agent

## Problem
55% of orders at a global e-commerce company arrive late.
This system predicts which orders are at risk and autonomously
investigates why — delivering a root cause memo without human intervention.

## Architecture
Layer 1 — Predict: XGBoost classifier (AUC ~0.90) on 180k orders
Layer 2 — Cluster: K-Means groups high-risk orders into actionable patterns
Layer 3 — Investigate: LangGraph agent runs regional/mode/seasonal analysis,
           routes to root cause, writes RCA memo via GPT-4o-mini

## Stack
Python · XGBoost · SHAP · scikit-learn · LangGraph · OpenAI · Streamlit · Power BI

## Dataset
DataCo Smart Supply Chain (Kaggle) — 180,519 records, 53 features, 2015-2019

## Setup
pip install -r requirements.txt
# Add OPENAI_API_KEY to .env
streamlit run app/streamlit_app.py

## Key Design Decisions
- Temporal train/test split (train 2015-2018, test 2019) — avoids temporal leakage
- Leakage removal: 'Days for shipping (real)' and 'Delivery Status' excluded
- SHAP TreeExplainer for per-order explainability feeding the agent context
- LangGraph over LangChain: conditional edge routing requires graph architecture
- GPT-4o-mini with constrained system prompt: prevents hallucinated recommendations


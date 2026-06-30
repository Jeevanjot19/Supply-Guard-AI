# SupplyGuard AI
### Autonomous Supply Chain Risk Investigation Agent

> *Predicts which shipments will be late. Investigates why. Tells operations exactly what to do — without human intervention.*

---

## What This Is

Most supply chain analytics stops at a risk score. A number between 0 and 1. Someone still has to open a dashboard, slice the data, write a report, and figure out what's actually wrong.

SupplyGuard AI does the next three steps autonomously.

It flags high-risk orders, groups them by the reason they're risky, and then runs a structured root cause investigation — regional analysis, shipping mode analysis, seasonal pattern check — before writing a business recommendation memo. The escalation or standard routing is decided by the agent itself based on what it finds.

Built on 180,519 real supply chain orders across 5 global markets. Every number in this repo is real and defensible.

---

## The Architecture — 3 Layers

```
Layer 1 — PREDICT
XGBoost classifier trained on order-placement-time features only.
Temporal split: train 2015–2016, test 2017.
Strict leakage removal — no post-shipment columns.
Output: risk probability per order (0–1)

         ↓

Layer 2 — CLUSTER
High-risk orders (predicted_risk > 0.6) extracted.
SHAP values computed for every order.
K-Means clustering on SHAP vectors — not raw features.
Groups orders by WHY they are risky, not what they look like.
Output: cluster profiles with dominant risk driver per group

         ↓

Layer 3 — INVESTIGATE
LangGraph agent picks up each cluster.
Runs 4 deterministic pandas analysis nodes:
  → Regional analysis
  → Shipping mode analysis
  → Seasonal pattern check
  → Profit impact calculation
LLM (Groq LLaMA) classifies root cause + severity.
Conditional edge: high severity → Escalation Memo
                  medium/low   → Standard Memo
Output: structured RCA memo with recommendations
```

---

## Key Findings

**Finding 1 — The First Class Paradox**

First Class shipping has the highest late delivery rate across virtually every region. Not because of fulfillment failure — because delivery windows are systematically over-promised. The model flags this clearly. The agent surfaces it in every relevant investigation.

**Finding 2 — The Model's Blind Spot**

Residual analysis on the model's missed predictions revealed that Standard Class shipping to Central America accounts for the largest share of all missed late deliveries. The model underestimates Standard Class risk because its loose promised window looks safe. This blind spot is explicitly embedded as an operational warning in the agent's mode analysis node — the system knows where it's wrong and says so.

**Finding 3 — Two Features Dominate Predictive Power**

`Days for shipment (scheduled)` and `Shipping Mode` carry the large majority of the model's predictive power. After removing leakage columns — verified through an explicit target-determinism check — the model reflects genuine order-placement-time signal rather than post-shipment information that leaks the outcome.

---

## Why SHAP-Based Clustering?

Standard clustering on raw features groups orders by what they look like. Two orders with identical shipping mode and region can be late for completely different reasons — one flagged because of the region, one because of the shipping mode.

SHAP values expose the actual risk driver per order. Clustering on SHAP vectors groups orders by why they're risky. Each cluster becomes a coherent investigation case — the agent knows before it starts which dimension to focus on because it's encoded in the cluster's dominant SHAP feature.

This is the core architectural decision that separates this from a standard late delivery prediction notebook.

---

## Why LangGraph and Not a Simple Pipeline?

A sequential pipeline runs the same steps for every input regardless of what the data shows. LangGraph's conditional edges mean the graph branches based on findings.

After the Router node classifies severity, a conditional edge routes to either an escalation memo (urgent, leads with profit figure, explicit action required) or a standard memo (structured, comprehensive). Two clusters with different severity levels follow different paths through the graph.

The analysis nodes (regional, mode, seasonal, profit) are deliberately deterministic pandas queries — you want exact numbers from real data, not LLM guesses. The LLM handles what it's actually good at: classification and natural language generation. This is the correct architecture for production analytics systems.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Risk Model | XGBoost + RandomizedSearchCV |
| Explainability | SHAP TreeExplainer |
| Clustering | K-Means on SHAP vectors |
| Agent Framework | LangGraph |
| LLM | Groq LLaMA 3.1 8B Instant |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit (interactive) + Power BI (executive) |
| Language | Python 3.11 |

---

## Project Structure

```
supplyguard-ai/
├── data/
│   ├── DataCoSupplyChainDataset.csv
│   ├── cluster_profiles.csv
│   ├── high_risk_orders.csv
│   └── powerbi_export.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_shap_analysis.ipynb
│   └── 05_clustering.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── cluster.py
│   └── agent/
│       ├── state.py
│       ├── nodes.py
│       ├── graph.py
│       └── prompts.py
├── models/
│   ├── xgb_model.pkl
│   ├── shap_explainer.pkl
│   ├── kmeans_model.pkl
│   └── feature_names.json
├── app/
│   └── streamlit_app.py
├── powerbi/
│   └── supplyguard_dashboard.pbix
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/jeevanjot/supplyguard-ai
cd supplyguard-ai
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

Get a free Groq API key at console.groq.com — no credit card required.

Download the dataset from Kaggle: [DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) and place it in `data/`.

---

## Running the Project

**Step 1 — EDA**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Step 2 — Train Model**
```bash
jupyter notebook notebooks/03_model_training.ipynb
# Use tune=False for quick run, tune=True for final model (~30 min)
```

**Step 3 — SHAP Analysis**
```bash
jupyter notebook notebooks/04_shap_analysis.ipynb
```

**Step 4 — Clustering**
```bash
jupyter notebook notebooks/05_clustering.ipynb
```

**Step 5 — Test Agent**
```bash
python quicktest.py
```

**Step 6 — Launch Streamlit App**
```bash
streamlit run app/streamlit_app.py
```

---

## Agent Output Sample

```
[Regional Analysis] Region 'Western Europe': late rate = 88.6% vs
overall 88.2% (delta +0.4%). Region is NOT a primary factor.

[Mode Analysis] Mode 'Second Class': late rate = 79.9% vs overall
88.2% (delta -8.3%). Mode IS a significant factor.

[Seasonal Analysis] Month 1: late rate = 87.5% vs overall 88.2%.
Seasonality is NOT a primary factor.

[Profit Impact] $721,963.82 gross order profit at risk across
34,945 flagged orders.

[Router] Primary cause: mode | Severity: high

[ESCALATION Memo] High-severity RCA memo generated.
```

---

## Design Decisions and Why

**Temporal split over random split**
Training on 2015–2016 and testing on 2017 mirrors production reality — you always predict the future from the past. Random splits leak future data into training and produce optimistic accuracy numbers.

**SHAP over built-in feature importance**
XGBoost's gain-based importance is unstable when colsample_bytree < 1.0 and misattributes importance among correlated features. SHAP handles correlation correctly and provides per-order local explanations — required for the clustering step and the Streamlit predictor.

**Groq LLaMA over OpenAI**
Same API interface, zero cost, no credit card. For a portfolio project running 50–100 agent invocations, cost should not be a constraint. The architecture is LLM-agnostic — swap the client in one line.

**Streamlit + Power BI (not one or the other)**
They serve different audiences. Streamlit is for running live investigations and showing the agent in action. Power BI is the business stakeholder dashboard — filterable, shareable, no Python required. Building both shows understanding of the difference between a data product and a BI product.

---

## Known Limitations

**Target is partially deterministic.** Determinism check showed 97.55% of `Late_delivery_risk` is derivable from the leakage columns. After removing them, AUC of 0.76 is the honest ceiling — not a model weakness.

**Dataset is synthetic.** DataCo Global is a fictional company. Real supply chains have more noise and more complex patterns.

**Agent has no cross-cluster memory.** Each cluster is investigated independently. A `detect_systemic_patterns()` function exists in `graph.py` for post-investigation aggregation but is not embedded in the live agent loop.

**Profit impact is gross order profit only.** Does not include refund costs, reshipment expenses, or customer lifetime value impact.

---

## What I Would Do Next

- Embed cross-cluster pattern detection into agent state for systemic issue flagging
- Add FastAPI inference endpoint for real-time order scoring
- Schedule nightly clustering via Airflow DAG
- Auto-trigger agent when cluster size exceeds threshold and email RCA memo to operations team

---

## Dataset

**DataCo Smart Supply Chain for Big Data Analysis**
Mendeley Data — DOI: 10.17632/8gx2fvg2k6.5
180,519 records | 53 features | 2015–2019
Markets: Europe, LATAM, Pacific Asia, USCA, Africa

---

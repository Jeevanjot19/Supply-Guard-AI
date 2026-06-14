
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.agent.graph import run_investigation

st.set_page_config(page_title='SupplyGuard AI', layout='wide', page_icon='🚚')

# ── Load artifacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model      = joblib.load(PROJECT_ROOT / 'models' / 'xgb_model.pkl')
    explainer  = joblib.load(PROJECT_ROOT / 'models' / 'shap_explainer.pkl')
    encoders   = joblib.load(PROJECT_ROOT / 'models' / 'encoders.pkl')
    feat_names = json.load(open(PROJECT_ROOT / 'models' / 'feature_names.json'))
    return model, explainer, encoders, feat_names

@st.cache_data
def load_data():
    df       = pd.read_csv(PROJECT_ROOT / 'data' / 'high_risk_orders.csv')
    profiles = pd.read_csv(PROJECT_ROOT / 'data' / 'cluster_profiles.csv')
    raw      = pd.read_csv(PROJECT_ROOT / 'data' / 'DataCoSupplyChainDataset.csv', encoding='unicode_escape')
    return df, profiles, raw

model, explainer, encoders, feat_names = load_artifacts()
high_risk_df, cluster_profiles, raw_df = load_data()

# ── Session state init ─────────────────────────────────────────────────
if 'agent_result' not in st.session_state:
    st.session_state.agent_result = None
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = 0

# ── Sidebar navigation ────────────────────────────────────────────────
page = st.sidebar.radio('Navigation', ['Overview', 'Risk Predictor', 'Agent Investigation'])
st.sidebar.markdown('---')
st.sidebar.markdown('**SupplyGuard AI** | DataCo Dataset')
st.sidebar.markdown('XGBoost + LangGraph + GPT-4o-mini')


# ═══════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == 'Overview':
    st.title('🚚 SupplyGuard AI — Supply Chain Risk Monitor')
    st.markdown('Autonomous late-delivery prediction and root cause investigation.')

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    total = len(raw_df)
    late_rate = raw_df['Late_delivery_risk'].mean()
    high_risk_count = len(high_risk_df)
    cluster_count = len(cluster_profiles)

    col1.metric('Total Orders', f'{total:,}')
    col2.metric('Late Delivery Rate', f'{late_rate:.1%}')
    col3.metric('High-Risk Flagged', f'{high_risk_count:,}')
    col4.metric('Risk Clusters Found', cluster_count)

    st.markdown('---')
    c1, c2 = st.columns(2)

    with c1:
        st.subheader('Late Rate by Shipping Mode')
        mode_data = raw_df.groupby('Shipping Mode')['Late_delivery_risk'].mean().reset_index()
        mode_data.columns = ['Shipping Mode', 'Late Rate']
        fig = px.bar(mode_data, x='Shipping Mode', y='Late Rate',
                     color='Late Rate', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader('Late Rate by Region (Top 10)')
        reg_data = raw_df.groupby('Order Region')['Late_delivery_risk'].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(reg_data, x='Late_delivery_risk', y='Order Region',
                     orientation='h', color='Late_delivery_risk', color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Cluster Risk Summary')
    st.dataframe(cluster_profiles, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2: RISK PREDICTOR
# ═══════════════════════════════════════════════════════════
elif page == 'Risk Predictor':
    st.title('Risk Predictor — Individual Order')
    st.markdown('Enter order details to get a late delivery risk score with explanation.')

    col1, col2 = st.columns(2)
    with col1:
        shipping_mode = st.selectbox('Shipping Mode',
            ['First Class', 'Second Class', 'Standard Class', 'Same Day'])
        order_region  = st.selectbox('Order Region',
            raw_df['Order Region'].unique().tolist())
        department    = st.selectbox('Department',
            raw_df['Department Name'].unique().tolist())
    with col2:
        month     = st.slider('Order Month', 1, 12, 6)
        quantity  = st.number_input('Order Quantity', 1, 100, 5)
        segment   = st.selectbox('Customer Segment',
            ['Consumer', 'Corporate', 'Home Office'])

    if st.button('Predict Risk', type='primary'):
        # Build feature vector matching feat_names
        # NOTE: In a real app, encode each field the same way preprocess.py did.
        # For demo, use a simplified lookup from the encoders dict.
        try:
            sm_enc  = encoders['Shipping Mode'].transform([shipping_mode])[0]
            reg_enc = encoders['Order Region'].transform([order_region])[0]
            dep_enc = encoders['Department Name'].transform([department])[0]
            seg_enc = encoders['Customer Segment'].transform([segment])[0]
            ms = np.sin(2 * np.pi * month / 12)
            mc = np.cos(2 * np.pi * month / 12)

            # Build a row matching the feature order — fill unknowns with 0
            row = pd.DataFrame([{f: 0 for f in feat_names}])
            row['Shipping Mode']    = sm_enc
            row['Order Region']     = reg_enc
            row['Department Name']  = dep_enc
            row['Customer Segment'] = seg_enc
            row['month_sin']        = ms
            row['month_cos']        = mc
            row['Order Item Quantity'] = quantity

            prob = model.predict_proba(row.values)[0][1]

            # Display gauge
            st.markdown(f'### Risk Score: {prob:.1%}')
            color = 'red' if prob > 0.7 else 'orange' if prob > 0.5 else 'green'
            fig = go.Figure(go.Indicator(
                mode='gauge+number', value=prob*100,
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': color},
                       'steps': [{'range': [0,50],'color':'lightgreen'},
                                 {'range': [50,70],'color':'lightyellow'},
                                 {'range': [70,100],'color':'lightsalmon'}]},
                title={'text': 'Late Delivery Risk %'}))
            st.plotly_chart(fig, use_container_width=True)

            # SHAP explanation
            import shap
            sv = explainer.shap_values(row.values)[0]
            top3_idx = np.argsort(np.abs(sv))[::-1][:3]
            st.markdown('**Top 3 Risk Factors (SHAP):**')
            for idx in top3_idx:
                direction = 'increases' if sv[idx] > 0 else 'decreases'
                st.write(f'• **{feat_names[idx]}** — {direction} risk by {abs(sv[idx]):.3f}')

        except Exception as e:
            st.error(f'Prediction error: {e}. Make sure encoders are fitted on these values.')


# ═══════════════════════════════════════════════════════════
# PAGE 3: AGENT INVESTIGATION
# ═══════════════════════════════════════════════════════════
elif page == 'Agent Investigation':
    st.title('Agent Investigation — Root Cause Analysis')
    st.markdown('Select a risk cluster. The agent will autonomously investigate and write a recommendation memo.')

    cluster_ids = cluster_profiles['cluster_id'].tolist()
    selected = st.selectbox('Select Cluster', cluster_ids,
        format_func=lambda x: f'Cluster {x} — {cluster_profiles[cluster_profiles["cluster_id"]==x]["dominant_region"].values[0]} / {cluster_profiles[cluster_profiles["cluster_id"]==x]["dominant_shipping_mode"].values[0]}')

    profile_row = cluster_profiles[cluster_profiles['cluster_id'] == selected].iloc[0].to_dict()

    st.markdown('**Cluster Profile:**')
    cols = st.columns(4)
    cols[0].metric('Orders', profile_row['order_count'])
    cols[1].metric('Avg Risk', f"{profile_row['avg_risk_score']:.1%}")
    cols[2].metric('Profit at Risk', f"${profile_row['profit_at_risk']:,.0f}")
    cols[3].metric('Dominant Mode', profile_row['dominant_shipping_mode'])

    if st.button('Run Agent Investigation', type='primary'):
        with st.spinner('Agent investigating... this takes 10-20 seconds...'):
            result = run_investigation(profile_row, str(PROJECT_ROOT / 'data' / 'high_risk_orders.csv'))
            st.session_state.agent_result = result

    if st.session_state.agent_result:
        result = st.session_state.agent_result

        st.markdown('---')
        st.subheader('Agent Trace')
        for step in result.get('trace', []):
            st.info(step)

        st.markdown('---')
        st.subheader('Root Cause Analysis Memo')
        st.markdown(result.get('recommendation_memo', 'No memo generated.'))

        # Download button
        st.download_button(
            label='Download Memo as .txt',
            data=result.get('recommendation_memo', ''),
            file_name=f'rca_cluster_{selected}.txt',
            mime='text/plain'
        )


import pandas as pd
import numpy as np
import json, os
from groq import Groq
from dotenv import load_dotenv
from src.agent.state import InvestigationState
from src.agent.prompts import ROUTER_SYSTEM, ROUTER_USER, MEMO_SYSTEM, MEMO_USER

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def _load_data(state: InvestigationState) -> pd.DataFrame:
    return pd.read_csv(state['full_data_path'])


def node_regional_analysis(state: InvestigationState) -> dict:
    df = _load_data(state)
    cluster_region = state['cluster_profile']['dominant_region']

    # Late delivery rate for this region vs rest
    in_region  = df[df['Order Region'] == cluster_region]['predicted_risk'].mean()
    out_region = df[df['Order Region'] != cluster_region]['predicted_risk'].mean()
    overall    = df['predicted_risk'].mean()

    delta = in_region - overall
    is_regional = abs(delta) > 0.05   # 5 percentage point threshold

    finding = (
        f"Region '{cluster_region}': late rate = {in_region:.1%} vs "
        f"overall {overall:.1%} (delta {delta:+.1%}). "
        f"{'Region IS a significant factor.' if is_regional else 'Region is NOT a primary factor.'}"
    )

    trace_entry = f'[Regional Analysis] {finding}'
    return {
        'regional_finding': finding,
        'trace': state.get('trace', []) + [trace_entry]
    }


def node_mode_analysis(state: InvestigationState) -> dict:
    df = _load_data(state)
    cluster_mode = state['cluster_profile']['dominant_shipping_mode']

    in_mode  = df[df['Shipping Mode'] == cluster_mode]['predicted_risk'].mean()
    overall  = df['predicted_risk'].mean()
    all_modes = df.groupby('Shipping Mode')['predicted_risk'].mean().to_dict()

    delta = in_mode - overall
    is_mode_driven = abs(delta) > 0.05

    mode_breakdown = ', '.join([f"{k}: {v:.1%}" for k, v in all_modes.items()])
    

    direction = "HIGHER" if in_mode > overall else "LOWER"
    finding = (
        f"Mode '{cluster_mode}': late rate = {in_mode:.1%} vs overall {overall:.1%} "
        f"(delta {delta:+.1%}, {direction} than average). "
        f"All modes: [{mode_breakdown}]. "
        f"{'Mode IS a significant factor.' if is_mode_driven else 'Mode is NOT a primary factor.'}"
    )

    trace_entry = f'[Mode Analysis] {finding}'
    return {
        'mode_finding': finding,
        'trace': state.get('trace', []) + [trace_entry]
    }


def node_seasonal_analysis(state: InvestigationState) -> dict:
    df = _load_data(state)
    cluster_month = state['cluster_profile']['dominant_month']

    if 'order_month' not in df.columns or cluster_month == 0:
        finding = 'Insufficient date data for seasonal analysis.'
        return {'seasonal_finding': finding,
                'trace': state.get('trace', []) + [f'[Seasonal Analysis] {finding}']
               }

    monthly_avg = df.groupby('order_month')['predicted_risk'].mean()
    overall = df['predicted_risk'].mean()
    this_month_rate = monthly_avg.get(cluster_month, df['predicted_risk'].mean())
    q4_rate = df[df['order_month'].isin([10, 11, 12])]['predicted_risk'].mean()
    delta = this_month_rate - overall
    is_seasonal = abs(delta) > 0.05

    # Check if Q4 months (10, 11, 12) are systematically higher
    # q4_rate = df[df['order_month'].isin([10, 11, 12])]['Late_delivery_risk'].mean()
    finding = (
        f"Month {cluster_month}: late rate = {this_month_rate:.1%} vs overall {overall:.1%} (delta {delta:+.1%}). "
        f"Q4 average: {q4_rate:.1%}. "
        f"{'Seasonality IS a factor.' if is_seasonal else 'Seasonality is NOT a primary factor.'}"
    )

    trace_entry = f'[Seasonal Analysis] {finding}'
    return {
        'seasonal_finding': finding,
        'trace': state.get('trace', []) + [trace_entry]
    }


def node_profit_impact(state: InvestigationState) -> dict:
    profit = state['cluster_profile'].get('profit_at_risk', 0.0)
    finding = f'${profit:,.2f} gross order profit at risk across {state["cluster_profile"]["order_count"]} flagged orders.'
    return {
        'profit_at_risk': float(profit),
        'trace': state.get('trace', []) + [f'[Profit Impact] {finding}']
    }


def node_router(state: InvestigationState) -> dict:
    prompt = ROUTER_USER.format(
        profile=json.dumps(state['cluster_profile']),
        regional=state.get('regional_finding', 'N/A'),
        mode=state.get('mode_finding', 'N/A'),
        seasonal=state.get('seasonal_finding', 'N/A'),
        profit=state.get('profit_at_risk', 0),
    )
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role': 'system', 'content': ROUTER_SYSTEM},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=80, temperature=0
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        cause    = parsed.get('primary_cause', 'combined')
        severity = parsed.get('severity', 'medium')
    except Exception:
        cause, severity = 'combined', 'medium'

    return {
        'primary_cause': cause, 'severity': severity,
        'trace': state.get('trace', []) + [f'[Router] Primary cause: {cause} | Severity: {severity}']
    }


def node_standard_memo(state: InvestigationState) -> dict:
    """Standard RCA memo for medium/low severity clusters."""
    prompt = MEMO_USER.format(
        profile=json.dumps(state['cluster_profile']),
        primary_cause=state.get('primary_cause', 'combined'),
        severity=state.get('severity', 'medium'),
        regional=state.get('regional_finding', 'N/A'),
        mode=state.get('mode_finding', 'N/A'),
        seasonal=state.get('seasonal_finding', 'N/A'),
        profit=state.get('profit_at_risk', 0),
    )
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role': 'system', 'content': MEMO_SYSTEM},
            {'role': 'user',   'content': prompt}
        ],
        max_tokens=600, temperature=0.3
    )
    memo = response.choices[0].message.content.strip()
    return {
        'recommendation_memo': memo,
        'trace': state.get('trace', []) + ['[Standard Memo] RCA memo generated.']
    }

def node_escalation_memo(state: InvestigationState) -> dict:
    """
    Escalation memo for high severity clusters.
    Leads with profit at risk, uses urgent language, shorter and more direct.
    """
    escalation_system = MEMO_SYSTEM + (
        '\n\nThis is a HIGH SEVERITY cluster. Lead with the profit at risk figure. '
        'Use direct, urgent language. Keep it under 300 words. '
        'End with a clear IMMEDIATE ACTION REQUIRED section.'
    )
    prompt = MEMO_USER.format(
        profile=json.dumps(state['cluster_profile']),
        primary_cause=state.get('primary_cause', 'combined'),
        severity='HIGH',
        regional=state.get('regional_finding', 'N/A'),
        mode=state.get('mode_finding', 'N/A'),
        seasonal=state.get('seasonal_finding', 'N/A'),
        profit=state.get('profit_at_risk', 0),
    )
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role': 'system', 'content': escalation_system},
            {'role': 'user',   'content': prompt}
        ],
        max_tokens=500, temperature=0.2   # lower temp = more urgent, less creative
    )
    memo = response.choices[0].message.content.strip()
    return {
        'recommendation_memo': memo,
        'trace': state.get('trace', []) + ['[ESCALATION Memo] High-severity RCA memo generated.']
    }

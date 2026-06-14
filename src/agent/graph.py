from langgraph.graph import StateGraph, START, END
from src.agent.state import InvestigationState
from src.agent.nodes import (
    node_regional_analysis, node_mode_analysis,
    node_seasonal_analysis, node_profit_impact,
    node_router, node_standard_memo, node_escalation_memo
)

def route_by_severity(state: InvestigationState) -> str:
    """
    Conditional edge function.
    Returns the name of the next node based on severity in state.
    This is what makes the graph genuinely branch — not a pipeline.
    """
    severity = state.get('severity', 'medium')
    if severity == 'high':
        return 'escalation_memo'
    return 'standard_memo'


def build_graph():
    builder = StateGraph(InvestigationState)

    # Analysis nodes — deterministic pandas queries
    builder.add_node('regional',        node_regional_analysis)
    builder.add_node('mode',            node_mode_analysis)
    builder.add_node('seasonal',        node_seasonal_analysis)
    builder.add_node('profit',          node_profit_impact)

    # LLM nodes
    builder.add_node('router',          node_router)
    builder.add_node('standard_memo',   node_standard_memo)
    builder.add_node('escalation_memo', node_escalation_memo)

    # Linear edges through analysis nodes
    builder.add_edge(START,      'regional')
    builder.add_edge('regional', 'mode')
    builder.add_edge('mode',     'seasonal')
    builder.add_edge('seasonal', 'profit')
    builder.add_edge('profit',   'router')

    # CONDITIONAL EDGE — this is what makes it a graph, not a pipeline
    builder.add_conditional_edges(
        'router',
        route_by_severity,
        {
            'standard_memo':   'standard_memo',
            'escalation_memo': 'escalation_memo',
        }
    )

    builder.add_edge('standard_memo',   END)
    builder.add_edge('escalation_memo', END)

    return builder.compile()


investigation_graph = build_graph()


def run_investigation(cluster_profile: dict, data_path: str) -> dict:
    initial_state = {
        'cluster_profile':      cluster_profile,
        'full_data_path':       data_path,
        'regional_finding':     '',
        'mode_finding':         '',
        'seasonal_finding':     '',
        'profit_at_risk':       0.0,
        'primary_cause':        '',
        'severity':             '',
        'recommendation_memo':  '',
        'trace':                [],
    }
    return investigation_graph.invoke(initial_state)


# def detect_systemic_patterns(all_results: list[dict]) -> str:
#     """
#     Run after all clusters are investigated.
#     Compares primary_cause across all results.
#     If the same cause appears in 3+ clusters, flags it as systemic.

#     Args:
#         all_results: list of result dicts from run_investigation()
#     Returns:
#         systemic_alert string (empty string if no pattern found)
#     """
#     from collections import Counter
#     causes = [r.get('primary_cause', 'unknown') for r in all_results]
#     counts = Counter(causes)

#     systemic = [(cause, count) for cause, count in counts.items() if count >= 3]

#     if not systemic:
#         return ''

#     alert_lines = ['=== SYSTEMIC PATTERN ALERT ===']
#     for cause, count in systemic:
#         total_profit = sum(
#             r.get('profit_at_risk', 0) for r in all_results
#             if r.get('primary_cause') == cause
#         )
#         alert_lines.append(
#             f'Root cause "{cause}" appears in {count} clusters. '
#             f'Combined profit at risk: ${total_profit:,.2f}. '
#             f'This is a SYSTEMIC issue, not a one-off event.'
#         )

#     return '\n'.join(alert_lines)


# # Usage — run after all clusters investigated:
# # all_results = [run_investigation(p.to_dict(), data_path) for _, p in profiles.iterrows()]
# # alert = detect_systemic_patterns(all_results)
# # if alert: print(alert)  # or display in Streamlit / email to ops team

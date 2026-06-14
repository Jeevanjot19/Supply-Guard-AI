from src.agent.graph import run_investigation
import pandas as pd

profiles = pd.read_csv('data/cluster_profiles.csv')

# Test one medium-severity cluster first
test_profile = profiles.iloc[0].to_dict()
result = run_investigation(test_profile, 'data/high_risk_orders.csv')

print('=== AGENT TRACE ===')
for step in result['trace']:
    print(step)
print()
print('=== FINAL MEMO ===')
print(result['recommendation_memo'])
print()
print(f'Severity: {result["severity"]} | Routed to: {"escalation" if result["severity"]=="high" else "standard"} memo')

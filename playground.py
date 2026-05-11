import pandas as pd
import json

# Load CSV
df = pd.read_csv("tests/analysis/advanced_metrics.csv")

# Convert first row to JSON
result = df.iloc[0].to_dict()

# Print pretty JSON
print(json.dumps(result, indent=4))
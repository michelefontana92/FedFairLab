import pandas as pd
from collections import Counter

df = pd.read_csv('data/Education/New/node_1/education_val.csv')
target ='SCHL'
value_counts = df[target].value_counts()
print(f"Value counts for column '{target}':")
for value, count in value_counts.items():
    print(f"{value}: {count} ({count / len(df) * 100:.2f}%)")
print(f"Total unique values in '{target}': {len(value_counts)}")
for col in df.columns:
    print(f"Column {col} unique values: {df[col].unique()}")
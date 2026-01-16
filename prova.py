import pandas as pd
from collections import Counter
def partition_schl(x):
    if x <= 15:
        return 'Low'
    elif x<=17:
        return 'High School'
    elif x <= 20:
        return 'Associate'
    else: 
        return 'Bachelor or higher'
    
df = pd.read_csv('data/Folktables_binary/New/node_1/folktables_binary_train.csv')
col = 'SCHL'
target_col = 'SCHL_CLASS'
value_counts = df[col].value_counts()
#print(f'Columns in the dataset: {df.columns.tolist()}')
df[target_col] = df[col].apply(partition_schl)
print(f"Value counts for column '{target_col}':")
for value, count in df[target_col].value_counts().items():
    print(f"{value}: {count} ({count / len(df) * 100:.2f}%)")
print(f"Total unique values in '{target_col}': {len(value_counts)}")


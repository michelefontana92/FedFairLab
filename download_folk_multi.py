from folktables import ACSDataSource, ACSIncome4
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
import os 

race = {
    1: "White",
    2: "Black",
    3: "Native",
    4: "Alaska",
    5: "AmericanIndian",
    6: "Asian",
    7: "Hawaiian",
    8: "Other",
    9: "TwoOrMore"
}

job = {
    1: "Employed Private profit",
    2: "Employed Private no profit",
    3: "Local government employee",
    4:  "State government employee",
    5: "Federal government employee",
    6: "Self-employed",
    7: "Self-employed incorporated",
    8: "Unpaid family worker",
    9: "Unemployed",
}

sex = {
    1:'Male',
    2: 'Female'
}

marital = {
    1: 'Married',
    2: 'Widowed',
    3: 'Divorced',
    4: 'Separated',
    5: 'Never Married',
}

state = 'AL'
def preprocess(state,node):
    data_source = ACSDataSource(survey_year='2014', horizon='1-Year', survey='person',root_dir='./data/Folktables_binary/Real')
    acs_data = data_source.get_data(states=[state], download=False)
    features, label, group = ACSIncome4.df_to_pandas(acs_data)
    df= pd.concat([features, label], axis=1)
    df['Race'] = df.apply(lambda x:race[int(x['RAC1P'])],axis=1)
    #df['Job'] = df.apply(lambda x:job[int(x['COW'])],axis=1)
    df['Gender'] = df.apply(lambda x: sex[int(x['SEX'])],axis=1)
    df['Marital'] = df.apply(lambda x: marital[int(x['MAR'])],axis=1)

    df['Race'] = df['Race'].apply(lambda x: 'Indigenous' if x in ['Native','Alaska','AmericanIndian','Hawaiian'] else x)
    df['Race'] = df['Race'].apply(lambda x: 'Other' if x in ['Other','TwoOrMore'] else x)

    #df['Job'] = df['Job'].apply(lambda x: 'Public Employee' if x in ['Local government employee','State government employee','Federal government employee'] else x)
    #df['Job'] = df['Job'].apply(lambda x: 'Self Employed' if x in ['Self-employed','Self-employed incorporated'] else x)
    #df['Job'] = df['Job'].apply(lambda x: 'Private Employee' if x in ['Employed Private profit','Employed Private no profit'] else x)
    #df['Job'] = df['Job'].apply(lambda x: 'Unemployed' if x in ['Unemployed','Unpaid family worker'] else x)
    #df = df[df['Job']!='Unemployed']
    df['Marital'] = df['Marital'].apply(lambda x: 'Other' if x in ['Widowed','Separated'] else x)
    #df = df.drop(columns=['SEX','RAC1P','COW','MAR'])
    df = df.drop(columns=['SEX','RAC1P','MAR'])
    #drop all records with Alaskian
    df = df[df['Race']!='Alaska']
    #stratify train-test split
    save_path = f'data/Income_4/Real/node_{node}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_csv(f'{save_path}/income_4_{state}_clean.csv', index=False)
    return df

def stratify_split(df, sensitive_attributes, target, save_path):
    print('Totale record iniziali:', len(df))

    # Crea label congiunto
    df['stratify_label'] = df[sensitive_attributes+ [target]].astype(str).agg('_'.join, axis=1)

    # Rimuovi gruppi troppo piccoli
    group_counts = df['stratify_label'].value_counts()
    small_groups = group_counts[group_counts <= 10].index
    df = df[~df['stratify_label'].isin(small_groups)]

    print(f"[✓] Rimosso {len(small_groups)} gruppi con <=30 record")
    print(f"[✓] Rimanenti per stratificazione: {len(df)}")

    # Split stratificato
    y = df[target]
    X = df.drop(columns=[target, 'stratify_label'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=df['stratify_label']
    )

    df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    df_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_train.to_csv(f'{save_path}/income_4_train.csv', index=False)
    df_test.to_csv(f'{save_path}/income_4_val.csv', index=False)

    print(f"[✓] Train: {len(df_train)} | Test: {len(df_test)}")


sensitive_attributes = ['Race','Marital']
target = 'PINCP'
save_path = 'data/Income_4/New'
#states = ['WV','AK','ND','PR','MS','AR','WI','WA','CT','LA','AL','MD','NY','TX','CA']
states = ['CA','NY','FL','TX','PA','IL','MD','GA','NC','MI']

for node,state in enumerate(states):
    print(f'Processing state {state}...')
    df = preprocess(state,node+1)
    if node == 0:
        df.to_csv(f'{save_path}/income_4_clean.csv', index=False)
    stratify_split(df, sensitive_attributes, target, f'{save_path}/node_{node+1}')
    print()
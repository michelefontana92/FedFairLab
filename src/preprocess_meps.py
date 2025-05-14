import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main(year,node):
    try:
        df = pd.read_sas(f'../data/Centralized_MEP/20{year}/meps_20{year}.ssp',format='xport')
    except: 
        df = pd.read_csv(f'../data/Centralized_MEP/20{year}/meps_20{year}.csv')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              f'POVCAT{year}' : 'POVCAT', f'INSCOV{year}' : 'INSCOV',"RACETHX" : "RACE"})

    
    df = df[df['REGION'] >= 0] # remove values -1
    df = df[df['AGE'] >= 0] # remove values -1
    df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9
    df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

    # Creazione della variabile target: spese sanitarie sopra la mediana
    df["HIGH_EXPENSES"] = (df[f"TOTEXP{year}"] > df[f"TOTEXP{year}"].quantile(0.75)).astype(int)

    print(df['HIGH_EXPENSES'].value_counts())
   
    df = df[['REGION','AGE','SEX','RACE','MARRY',
                                    'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                    'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                    'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                    'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                    'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','HIGH_EXPENSES', 
                                    f'PERWT{year}F']]


    target = 'HIGH_EXPENSES'

    df['AGE_CAT'] = pd.qcut(df['AGE'], q=4, labels=['Young', 'Mid_Age', 'Senior', 'Elderly'])

    df['MARRY'] = df['MARRY'].replace({1:'Married',
                                    2:'Widowed',
                                    3:'Divorced',
                                    4:'Separated',
                                    5:'Never Married',
                                    6:'Under 16',
                                    7: 'Married in Round',
                                    8: 'Widowed in Round',
                                    9: 'Divorced in Round',
                                    10: 'Separated in Round'})

    df['SEX'] = df['SEX'].replace({1:'Male',2:'Female'})

    df['REGION'] = df['REGION'].replace({1:'Northeast',
                                        2:'Midwest',
                                        3:'South',
                                        4:'West'})
    df['RACE'] = df['RACE'].replace({1:'Hispanic',
                                    2:'White',
                                    3:'Black',
                                    4:'Asian',
                                    5:'Other'})

    df['MARRY'] = df['MARRY'].apply(lambda x: 'Other' if x in [
                                 'Married in Round',
                                 'Widowed in Round',
                                  'Divorced in Round',
                                  'Separated in Round',
                                  'Separated',
                                  'Widowed',
                                  'Divorced'] else x)

    df['MARRY'] = df['MARRY'].apply(lambda x: 'Never Married' if x == 'Under 16' else x)
    df['RACE'] = df['RACE'].apply(lambda x: 'Other' if x not in ['White','Black','Hispanic'] else x)
    
    sensitive_attributes=['SEX','RACE','MARRY']
    target='HIGH_EXPENSES'
    # Creazione di una colonna combinata per la stratificazione
    df['stratify_label'] = df[sensitive_attributes].astype(str).agg('_'.join, axis=1)  # Concatena tutte le colonne sensibili
    joint_distribution_full = pd.crosstab(df['stratify_label'], df[target], normalize=False)
    print("Distribuzione congiunta nel dataset originale:\n", joint_distribution_full)
    # Separazione X e y
    y = df[target]
    X = df.drop(columns=[target, 'stratify_label'])  # Rimuove la colonna dopo averla usata per stratificare

    # Stratificazione usando la colonna combinata
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=df['stratify_label']
    )

    df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    df_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    path = f'../data/Centralized_MEP/node_{node}'
    # Create the directory if it doesn't exist
    
    os.makedirs(path, exist_ok=True)
    df_train.to_csv(f'{path}/mep_train.csv', index=False)
    df_test.to_csv(f'{path}/mep_val.csv', index=False)

    #for col in df.columns:
    #    print(f'Column {col}: Unique values: {df[col].unique()}')

    print(df_train.columns)
if __name__ == '__main__':
    #df = pd.read_csv('../data/Centralized_MEP/node_1/mep_train.csv')
    #print(df.columns)

    #df = pd.read_excel('../data/Centralized_MEP/2017/meps_2017.xlsx',engine="openpyxl")
    #df = pd.read_csv('../data/Centralized_MEP/2017/meps_2017.csv', sep=';', encoding='utf-8-sig', low_memory=False)
    #print(df.columns[[568, 1085, 1561]])
    #df.to_csv('../data/Centralized_MEP/2017/meps_2017.csv',index=False)
    main(year=17,node=3)
    """
    for node,year in enumerate(range(16,20)):
        print(f'node {node} year {year}')
        main(year=year,node=node+2)
    """
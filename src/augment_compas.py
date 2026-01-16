from gan import CTABGAN
import numpy as np
import pandas as pd
import pickle as pkl


if __name__ == '__main__':
    N = 100000
    with open('../data/FL_Compas/gan_compas.p', 'rb') as f:
        gan = pkl.load(f)
    data:pd.DataFrame = gan.generate_samples(N)
    df = pd.read_csv("../data/Centralized_Compas/compas_clean.csv")
    data = pd.concat([df, data], ignore_index=True).reset_index(drop=True)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f'[✓] Generated {len(data)} samples after removing duplicates')
    data.to_csv("../data/FL_Compas/compas_augmented.csv",
                index=False)

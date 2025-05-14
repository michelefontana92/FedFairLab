from gan import CTABGAN
import numpy as np
import pandas as pd
import pickle as pkl


if __name__ == '__main__':
    N = 500000
    with open('../data/Centralized_MEP/gan_mep.p', 'rb') as f:
        gan = pkl.load(f)
    data = gan.generate_samples(N)
    data.to_csv("../data/Centralized_MEP/mep_augmented.csv",
                index=False)

from gan import CTABGAN
import pickle as pkl

if __name__ == '__main__':
    real_path = "../data/Centralized_MEP/mep1_clean.csv"
    fake_file_root = "../data/Centralized_MEP/mep1_fake.csv"
    synthesizer = CTABGAN(raw_csv_path=real_path,
                          test_ratio=0.20,
                          categorical_columns=[
            'RACE',
            'SEX',
            'MARRY',
            'AGE_CAT',
            'REGION',
            'FTSTU',
            'ACTDTY',
            'HONRDC',
            'RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','MIDX',
            'OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX',
            'DIABDX','JTPAIN','ARTHDX','ARTHTYPE','ASTHDX',
            'ADHDADDX','PREGNT','WLKLIM','ACTLIM','SOCLIM',
            'COGLIM','DFHEAR42','DFSEE42','ADSMOK42','EMPST',
            'POVCAT','INSCOV','HIGH_EXPENSES'
        ],
                          log_columns=[],
                          mixed_columns={},
                          integer_columns=[
            'AGE','PCS42','MCS42','K6SUM42','PHQ242','PERWT15F'
        ],
                          problem_type={"Classification": 'HIGH_EXPENSES'},
                          epochs=150
                          )

    synthesizer.fit()
    with open('../data/Centralized_MEP/gan_mep.p', 'wb') as f:
        pkl.dump(synthesizer, f)

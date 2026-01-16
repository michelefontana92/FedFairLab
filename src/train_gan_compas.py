from gan import CTABGAN
import pickle as pkl
import os 
if __name__ == '__main__':
    real_path = '../data/Centralized_Compas/compas_clean.csv'
    fake_file_root = '../data/FL_Compas/compas_fake.csv'
    if not os.path.exists(fake_file_root):
        os.makedirs(fake_file_root)
    synthesizer = CTABGAN(raw_csv_path=real_path,
                          test_ratio=0.20,
                          categorical_columns=[
            'c_charge_degree',
            'age_cat',
            'score_text',
            'decile_score',
            'sex',
            'race','two_year_recid'
        ],
                          log_columns=[],
                          mixed_columns={},
                          integer_columns=[
            'age',
            'priors_count',
            'days_b_screening_arrest',
            'length_of_stay',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
            'c_days_from_compas'
        ],
                          problem_type={"Classification": 'two_year_recid'},
                          epochs=1000
                          )

    synthesizer.fit()
    with open('../data/FL_Compas/gan_compas.p', 'wb') as f:
        pkl.dump(synthesizer, f)

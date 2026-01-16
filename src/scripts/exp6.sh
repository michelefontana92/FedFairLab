#python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedavg -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedavg -e New --num_classes 3 -ml demographic_parity  -gl Marital -tl 0.20 -ml demographic_parity  -gl Race -tl 0.20
##python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_race_020 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Race -tl 0.20 -a fedfairlab -e New --num_classes 3
#python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_marital_020 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Marital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
#python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_marital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Marital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
#python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_race_020_pb06 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Race -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60

#python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_race_010 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Race -tl 0.10 -a fedfairlab -e New --num_classes 3 -pb 0.60
#python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_marital_010 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Marital -tl 0.10 -a fedfairlab -e New --num_classes 3 -pb 0.60

#python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_marital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Marital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
#python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_race_010 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Race -tl 0.10 -a fedfairlab -e New --num_classes 3 -pb 0.60
#python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_marital_010 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Marital -tl 0.10 -a fedfairlab -e New --num_classes 3 -pb 0.60

#python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_eod_marital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl Marital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_eod_new_race_010 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds -gl Race -tl 0.10 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_eod_new_marital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl Marital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_dp_new_race_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity -gl Race -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_dp_new_marital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Marital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60

#python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_eod_race_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl Race -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60

#python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_eod_marital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl Marital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_eod_new_race_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl Race -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_eod_new_marital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl Marital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_dp_new_race_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Race -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_dp_new_marital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl Marital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60

python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_dp_new_racemarital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl RaceMarital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_dp_new_gendermarital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl GenderMarital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_dp_new_genderrace_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl GenderRace -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60

python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_eod_new_racemarital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl RaceMarital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_eod_new_gendermarital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl GenderMarital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r income4_fedfairlab  -p Income_3_Multiclass -i fedfairlab_eod_new_genderrace_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl GenderRace -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60

python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_dp_new_racemarital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl RaceMarital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_dp_new_gendermarital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl GenderMarital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_dp_new_genderrace_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml demographic_parity  -gl GenderRace -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60

python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_eod_new_racemarital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl RaceMarital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_eod_new_gendermarital_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl GenderMarital -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60
python main.py -r education_fedfairlab  -p Education_Multiclass -i fedfairlab_eod_new_genderrace_020 -ng 3 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -ml equalized_odds  -gl GenderRace -tl 0.20 -a fedfairlab -e New --num_classes 3 -pb 0.60

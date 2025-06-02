python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_marital_010 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl Marital -ml demographic_parity -tl 0.10 -pb 0.74
python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_job_010 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl Job -ml demographic_parity -tl 0.10 -pb 0.74

python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_race_020 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl Race -ml demographic_parity -tl 0.20 -pb 0.74
python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_marital_020 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl Marital -ml demographic_parity -tl 0.20 -pb 0.74

python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_race_marital_pb -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl RaceMarital -ml demographic_parity -tl 0.20 -pb 0.74

python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_race_marital_eod -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl RaceMarital -ml equalized_odds -tl 0.20 -pb 0.74
python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_job_race_eod -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl JobRace -ml equalized_odds -tl 0.20 -pb 0.74
python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_job_marital_eod -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl JobMarital -ml equalized_odds -tl 0.20 -pb 0.74


python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_race_eod_020 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl Race -ml equalized_odds -tl 0.20 -pb 0.74
python main.py -r folk_fedfairlab  -p FedFairLab_Folk_New -i fedfairlab_marital_eod_020 -ng 1 -nl 5 -lp 5 -nf 30 -np 100 -nc 10 -a fedfairlab -e New -gl Marital -ml equalized_odds -tl 0.20 -pb 0.74
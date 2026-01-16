python main.py -r employment_fedfairlab  -p FedDist05_Employment -i fedavg_05 -ng 1 -nl 5 -lp 5 -nf 100 -np 100 -nc 10 -a fedavg -e Dirichlet_05
python main.py -r compas_fedfairlab  -p FedDist05_Compas -i fedavg_05 -ng 1 -nl 5 -lp 5 -nf 100 -np 100 -nc 10 -a fedavg -e Dirichlet_05
python main.py -r mep_fedfairlab  -p FedDist05_MEPS -i fedavg_05 -ng 1 -nl 5 -lp 5 -nf 100 -np 100 -nc 10 -a fedavg -e Dirichlet_05

python main.py -r compas_fedfairlab  -p FedDist05_Compas -i fedfairlab_05 -ng 1 -nl 5 -lp 5 -nf 100 -np 100 -nc 10 -a fedfairlab -e Dirichlet_05

python main.py -r mep_fedfairlab  -p FedDist05_MEPS -i fedfairlab_05 -ng 1 -nl 5 -lp 5 -nf 100 -np 100 -nc 10 -a fedfairlab -e Dirichlet_05

python main.py -r mep_fedfairlab  -p FedDist05_MEPS -i fedavg_05 -ng 1 -nl 5 -lp 5 -nf 100 -np 100 -nc 10 -a fedavg -e Dirichlet_05

python main.py -r folk_fedfairlab  -p FedDist05_Income -i fedfairlab_05 -ng 1 -nl 5 -lp 5 -nf 100 -np 100 -nc 10 -a fedfairlab -e Dirichlet_05
python main.py -r employment_fedfairlab  -p FedDist05_Employment -i fedfairlab_05 -ng 1 -nl 5 -lp 5 -nf 100 -np 100 -nc 10 -a fedfairlab -e Dirichlet_05

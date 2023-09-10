#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: bash run_all.sh SPLIT NODE"
    exit
fi
split=$1
node=$2

family_base="26_Sent140_tune_I_4_H_0.7_N_8_argo"
dataset=Sent140
N=8
R=2000
I=4
H=0.7
alg="local_clip"

etas=("0.01" "0.03" "0.1" "0.3")
gammas=("0.003" "0.009" "0.03" "0.09")

j=$((2*$split+$node))
eta=${etas[$j]}
gamma=${gammas[$j]}
bash run.sh ${family_base}/0_S_2 eta_${eta} $dataset $alg $N 2 $R $I $H $eta $gamma 1 0 2 0
bash run.sh ${family_base}/1_S_4 eta_${eta} $dataset $alg $N 4 $R $I $H $eta $gamma 1 0 4 0

j=$split
eta=${etas[$j]}
gamma=${gammas[$j]}
bash run.sh ${family_base}/2_S_6 eta_${eta} $dataset $alg $N 6 $R $I $H $eta $gamma 2 $node 3 0
bash run.sh ${family_base}/3_S_8 eta_${eta} $dataset $alg $N 8 $R $I $H $eta $gamma 2 $node 4 0

j=$(($split+2))
eta=${etas[$j]}
gamma=${gammas[$j]}
bash run.sh ${family_base}/2_S_6 eta_${eta} $dataset $alg $N 6 $R $I $H $eta $gamma 2 $node 3 0
bash run.sh ${family_base}/3_S_8 eta_${eta} $dataset $alg $N 8 $R $I $H $eta $gamma 2 $node 4 0

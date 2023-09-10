#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: bash effect_of_S.sh SPLIT NODE"
    exit
fi
split=$1
node=$2

family_idx=31
dataset=SNLI
N=8
R=5375
I=4
H=0.7
eta=0.03
gamma=0.03
algs=("local_clip" "episode_mem" "episode" "naive_parallel_clip" "scaffold_clip" "minibatch_clip")
family_base="${family_idx}_${dataset}_I_${I}_H_${H}_N_${N}_effect_of_S_argo"

j=$((2*$split+$node))
alg=${algs[$j]}
bash run.sh ${family_base}/0_S_2 $alg $dataset $alg $N 2 $R $I $H $eta $gamma 1 0 2 0
bash run.sh ${family_base}/1_S_4 $alg $dataset $alg $N 4 $R $I $H $eta $gamma 1 0 4 0
if [ "$split" == 0 ]; then
    extra=$(($j+4))
    alg=${algs[$extra]}
    bash run.sh ${family_base}/0_S_2 $alg $dataset $alg $N 2 $R $I $H $eta $gamma 1 0 2 0
    bash run.sh ${family_base}/1_S_4 $alg $dataset $alg $N 4 $R $I $H $eta $gamma 1 0 4 0
fi

j=$((2*$split))
alg=${algs[$j]}
bash run.sh ${family_base}/2_S_6 $alg $dataset $alg $N 6 $R $I $H $eta $gamma 2 $node 3 0
bash run.sh ${family_base}/3_S_8 $alg $dataset $alg $N 8 $R $I $H $eta $gamma 2 $node 4 0

j=$(($j+1))
alg=${algs[$j]}
bash run.sh ${family_base}/2_S_6 $alg $dataset $alg $N 6 $R $I $H $eta $gamma 2 $node 3 0
bash run.sh ${family_base}/3_S_8 $alg $dataset $alg $N 8 $R $I $H $eta $gamma 2 $node 4 0

if [ "$split" == 0 ]; then
    extra=$(($j+3))
    alg=${algs[$extra]}
    bash run.sh ${family_base}/2_S_6 $alg $dataset $alg $N 6 $R $I $H $eta $gamma 2 $node 3 0
    bash run.sh ${family_base}/3_S_8 $alg $dataset $alg $N 8 $R $I $H $eta $gamma 2 $node 4 0

    extra=$(($extra+1))
    alg=${algs[$extra]}
    bash run.sh ${family_base}/2_S_6 $alg $dataset $alg $N 6 $R $I $H $eta $gamma 2 $node 3 0
    bash run.sh ${family_base}/3_S_8 $alg $dataset $alg $N 8 $R $I $H $eta $gamma 2 $node 4 0
fi

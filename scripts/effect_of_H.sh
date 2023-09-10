#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: bash effect_of_H.sh NODE"
    exit
fi
node=$1

family_base="29_Sent140_I_4_N_8_S_4_effect_of_H_argo"
dataset=Sent140
N=8
S=4
R=2000
I=4
eta=0.03
gamma=0.01
algs=("local_clip" "episode_mem" "episode" "naive_parallel_clip" "scaffold_clip" "minibatch_clip")

alg=${algs[$node]}
bash run.sh ${family_base}/0_H_0.8 $alg $dataset $alg $N $S $R $I 0.8 $eta $gamma 1 0 4 0
bash run.sh ${family_base}/2_H_1.0 $alg $dataset $alg $N $S $R $I 1.0 $eta $gamma 1 0 4 0

if [ "$node" == 0 ] || [ "$node" == 1 ]; then
    extra=$(($node+4))
    alg=${algs[$extra]}
    bash run.sh ${family_base}/0_H_0.8 $alg SNLI $alg $N $S $R $I 0.8 $eta $gamma 1 0 4 0
    bash run.sh ${family_base}/2_H_1.0 $alg SNLI $alg $N $S $R $I 1.0 $eta $gamma 1 0 4 0
fi

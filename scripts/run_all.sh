#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: bash run_all.sh NODE"
    exit
fi
node=$1

family_base="24_SNLI_I_4_N_8_S_4_effect_of_H_argo"
N=8
S=4
R=5375
I=4
eta=0.03
gamma=0.03
algs=("local_clip" "episode_mem" "episode" "naive_parallel_clip" "scaffold_clip" "minibatch_clip")

H=0.7

alg=${algs[$node]}
bash run.sh ${family_base}/0_H_0.5 $alg SNLI $alg $N $S $R $I 0.5 $eta $gamma 1 0 4 0
bash run.sh ${family_base}/2_H_0.9 $alg SNLI $alg $N $S $R $I 0.9 $eta $gamma 1 0 4 0

if [ "$node" == 0 ] || [ "$node" == 1 ]; then
    extra=$(($node+4))
    alg=${algs[$extra]}
    bash run.sh ${family_base}/0_H_0.5 $alg SNLI $alg $N $S $R $I 0.5 $eta $gamma 1 0 4 0
    bash run.sh ${family_base}/2_H_0.9 $alg SNLI $alg $N $S $R $I 0.9 $eta $gamma 1 0 4 0
fi


#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: bash effect_of_H.sh NODE"
    exit
fi
node=$1

family_base="30_SNLI_H_0.7_N_8_S_4_effect_of_I_argo"
dataset=SNLI
N=8
S=4
H=0.7
eta=0.03
gamma=0.01
algs=("local_clip" "episode_mem" "episode" "naive_parallel_clip" "scaffold_clip" "minibatch_clip")

Is=("2" "8" "16")
Rs=("10750" "2688" "1344")
idxs=("0" "2" "3")

alg=${algs[$node]}
for i in {0..2}; do
    I=${Is[$i]}
    R=${Rs[$i]}
    idx=${idxs[$i]}
    bash run.sh ${family_base}/${idx}_I_${I} $alg $dataset $alg $N $S $R $I $H $eta $gamma 1 0 4 0
done

if [ "$node" == 0 ] || [ "$node" == 1 ]; then
    extra=$(($node+4))
    alg=${algs[$extra]}
    for i in {0..2}; do
        I=${Is[$i]}
        R=${Rs[$i]}
        idx=${idxs[$i]}
        bash run.sh ${family_base}/${idx}_I_${I} $alg $dataset $alg $N $S $R $I $H $eta $gamma 1 0 4 0
    done
fi

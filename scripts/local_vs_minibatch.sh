#!/bin/bash

TOTAL_CLIENTS=32
TOTAL_NODES=1
NODE=0
WORKERS_PER_NODE=8
I=8
ETA=0.1
GAMMA=0.1
START_IDX=6

algs=("local_clip" "minibatch_clip")
participating=("32" "8")
pnames=("full" "partial")
Hs=("0.0" "0.75")
hnames=("homo" "hetero")

for i in {0..1}; do
    for j in {0..1}; do
        idx=$(($START_IDX + 2 * $i + $j))
        pname=${pnames[$i]}
        hname=${hnames[$j]}
        name=${idx}_${pname}_${hname}
        p=${participating[$i]}
        h=${Hs[$j]}
        for k in {0..1}; do
            alg=${algs[$k]}
            bash run.sh $name $alg CIFAR10 $alg $TOTAL_CLIENTS $p $TOTAL_NODES $NODE $WORKERS_PER_NODE $I $h $ETA $GAMMA
        done
    done
done

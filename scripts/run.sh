#!/bin/bash

if [ "$#" -ne 15 ]; then
    echo "Usage: bash run.sh FAMILY NAME DATASET ALG TOTAL_CLIENTS PARTICIPATING_CLIENTS R I H ETA GAMMA TOTAL_NODES NODE GPUS_PER_NODE START_GPU"
    exit
fi

FAMILY=$1
NAME=$2
DATASET=$3
ALG=$4
TOTAL_CLIENTS=$5
PARTICIPATING_CLIENTS=$6
R=$7
I=$8
H=$9
ETA=${10}
GAMMA=${11}
TOTAL_NODES=${12}
NODE=${13}
GPUS_PER_NODE=${14}
START_GPU=${15}
WORLD_SIZE=$(($GPUS_PER_NODE * $TOTAL_NODES))
BASE_DIR=../logs/${FAMILY}
LOG_DIR=${BASE_DIR}/${NAME}

mkdir -p $BASE_DIR
mkdir -p $LOG_DIR

SHAREDFILE="file://"$(readlink -f ${LOG_DIR})"/sharedfile"
if [ "$ALG" == "fedavg" ]; then
    ALG="local_clip"
    extra=""
    GAMMA=1e8
elif [ "$ALG" == "local_clip" ]; then
    extra=""
elif [ "$ALG" == "naive_parallel_clip" ]; then
    ALG="minibatch_clip"
    R=$(($R * $I))
    I=1
elif [ "$ALG" == "minibatch_sgd" ]; then
    ALG="minibatch_clip"
    extra=""
    GAMMA=1e8
elif [ "$ALG" == "minibatch_clip" ]; then
    extra=""
elif [ "$ALG" == "scaffold" ]; then
    extra="--init_corrections"
    GAMMA=1e8
elif [ "$ALG" == "scaffold_clip" ]; then
    ALG="scaffold"
    extra="--init_corrections"
elif [ "$ALG" == "episode" ]; then
    extra=""
elif [ "$ALG" == "episode_mem" ]; then
    extra="--init_corrections"
elif [ "$ALG" == "fedprox" ]; then
    ALG="local_clip"
    extra="--fedprox --fedprox-mu 0.01"
    GAMMA=1e8
else
    echo "Unrecognized algorithm: $ALG."
    exit
fi

if [ "$DATASET" == "CIFAR10" ]; then
    loss="svm"
    model="resnet32"
    batch_size=128
    num_evals=30
elif [ "$DATASET" == "MNIST" ]; then
    loss="svm"
    model="resnet18"
    batch_size=128
    num_evals=20
elif [ "$DATASET" == "SNLI" ]; then
    loss="svm"
    model="rnn"
    batch_size=64
    num_evals=30
    extra="${extra} --encoder_dim 2048 --n_enc_layers 1 --rnn"
elif [ "$DATASET" == "Sent140" ]; then
    loss="svm"
    model="rnn"
    batch_size=64
    num_evals=30
    extra="${extra} --encoder_dim 2048 --n_enc_layers 1 --rnn"
elif [ "$DATASET" == "ShakeSpeare" ]; then
    loss="cross_entropy"
    model="rnn"
    batch_size=64
    num_evals=10 # TEMP
    extra="${extra} --encoder_dim 2048 --n_enc_layers 1 --rnn"
else
    echo "Unrecognized dataset: $DATASET."
    exit
fi
milestones="$(($R / 2)) $(($R * 3 / 4))"

i=0
pids=""
while [ $i -lt $GPUS_PER_NODE ]; do

    rank=$(($NODE * $GPUS_PER_NODE + $i))
    gpu=$(($START_GPU + $i))
    python3 ../main.py \
        --init-method $SHAREDFILE \
        --model $model \
        --loss $loss \
        --eta0 $ETA \
        --weight-decay 5e-4 \
        --step-decay-milestones $milestones \
        --step-decay-factor 0.5 \
        --clipping-param $GAMMA \
        --algorithm $ALG \
        --total-clients $TOTAL_CLIENTS \
        --participating-clients $PARTICIPATING_CLIENTS \
        --world-size $WORLD_SIZE \
        --rank $rank \
        --gpu-id $gpu \
        --communication-interval $I \
        --rounds $R \
        --num-evals $num_evals \
        --batchsize $batch_size \
        --dataset $DATASET \
        --dataroot ../data \
        --reproducible \
        --seed 0 \
        --heterogeneity $H \
        --log-folder $LOG_DIR \
        --init-model $BASE_DIR/init_model.pth \
        $extra \
        > ${LOG_DIR}/worker_${rank}.out &

    pids="${pids} $!"
    i=$(($i + 1))
done

echo "children:${pids}"
wait

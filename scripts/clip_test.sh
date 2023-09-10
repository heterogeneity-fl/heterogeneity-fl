#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: bash clip_test.sh NODE"
    exit
fi
node=$1

eta=0.1
gammas=("0.01" "0.03" "0.1" "0.3")

gamma=${gammas[$node]}
bash run.sh 25_clip_test_Sent140 eta_${eta}_gamma_${gamma} Sent140 local_clip 8 4 25 4 0.7 $eta $gamma 1 0 4 0

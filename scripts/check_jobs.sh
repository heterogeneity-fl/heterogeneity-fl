if [ $(hostname | cut -c 1-6) == "hopper" ]; then
    partition=gpu-test
    key=" R "
elif [ $(hostname | cut -c 1-4) == "ARGO" ]; then
    partition=gpuq
    key=NODE
else
    echo "Invalid cluster."
    exit
fi

#cluster=$1
#if [ $cluster == "argo" ]; then
#    partition=gpuq
#    key=NODE
#elif [ $cluster == "hopper" ]; then
#    partition=gpu-test
#    key=" R "
#else
#    echo "Invalid cluster."
#    exit
#fi

hosts=$(squeue -p $partition | grep mcrawsha | grep "$key" | awk '{print $8}')
for host in $hosts
do
    echo $host
    ssh $host nvidia-smi
    printf "\n\n"
done

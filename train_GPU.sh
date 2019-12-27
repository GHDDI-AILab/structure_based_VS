#!/usr/bin/env bash

#$ -N train_GAT
#$ -cwd
#$ -l ngpus=1
#$ -q p100
source /home/cudasoft/bin/startcuda.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
conda activate py36
python train.py -device 1
source /home/cudasoft/bin/end_cuda.sh

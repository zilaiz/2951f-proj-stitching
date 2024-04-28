#!/bin/bash

# Request specific runtime (default for partition as per sinfo is not valid). Max allowed: 24h
#SBATCH --time=08:00:00

# Request a GPU partition node and access to 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:1 --exclude gpu2103,gpu2107,gpu2114,gpu2115,gpu2116
###SBATCH --partition=gpu -C a5000 --gres=gpu:1 --exclude gpu2103,gpu2107,gpu2114,gpu2115,gpu2116
#SBATCH --mem=24G
#SBATCH -n 4
#SBATCH -N 1

# source /users/zzeng28/.bashrc
source ~/.bashrc
export LC_ALL=en_US.utf8
export LANG=en_US.utf8

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# module load anaconda
module load mesa/22.1.6-6dbg5gq
module load tmux/3.3a-zyhjvvh
module load patchelf/0.17.2-aqmx4qb
module load glew/2.2.0-plawm2j

source activate stitch

cd /oscar/data/csun45/yluo73/r2024/2951f-proj-stitching/

while getopts d:a:n:p:b:l:s:u:i: flag
do
    case "${flag}" in
        d) dataset_name=${OPTARG};;
        a) is_augment=${OPTARG};;
        n) nclusters=${OPTARG};;
        p) augment_prob=${OPTARG};;
        b) batch_size=${OPTARG};;
        l) lr=${OPTARG};;
        s) seed=${OPTARG};;
        u) num_updates_per_iter=${OPTARG};;
        i) max_iters=${OPTARG};;
    esac
done


# python3 train_dmlp.py \
#     dataset_name=${dataset_name} \
#     augment_data=${is_augment} \
#     nclusters=${nclusters} \
#     augment_prob=${augment_prob} \
#     batch_size=${batch_size} lr=${lr} \
#     seed=${seed} \
#     num_updates_per_iter=${num_updates_per_iter} \
#     max_train_iters=${max_iters}

#bash $FILE $@
{
# config="cfgs/ben/ben_pointMmaze_aug05.yaml"
config="cfgs/ben/ben_pointMmaze_aug05OnlyGoal.yaml"
cp="../$(dirname $config)"
cn="$(basename $config)"
cn="${cn%.*}" # remove string after .
echo $cp $cn # echo $base_path
# HYDRA_FULL_ERROR=1 \
python3 scripts/train_dmlp_luo.py \
        -cp $cp -cn $cn +cfg_path=$config \
        dataset_name=${dataset_name} \
        augment_data=${is_augment} \
        nclusters=${nclusters} \
        augment_prob=${augment_prob} \
        batch_size=${batch_size} lr=${lr} \
        seed=${seed} \
        num_updates_per_iter=${num_updates_per_iter} \
        max_train_iters=${max_iters} \
        # augment_data=False \
        # nclusters=40 \
        # wandb_log=False \
        # num_workers=0 batch_size=16 wandb_log=False \
        
}
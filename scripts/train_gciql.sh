#!/bin/bash
source ~/.bashrc
source activate stitch
alias python=python3
module load mesa

config="cfgs/gciql_luotest.yaml"
config="cfgs/gciql_antUmaze.yaml"
# config="cfgs/gciql_antLmaze.yaml"

config="cfgs/gciql_pointLmaze.yaml"

config="cfgs/gciql_antUmaze_rlkit.yaml"
config="cfgs/gciql/gciql_pointMmaze_rlkit_aug05_minTogoal10.yaml"

cp="../$(dirname $config)"
cn="$(basename $config)"
cn="${cn%.*}" # remove string after .
echo $cp $cn # echo $base_path

# HYDRA_FULL_ERROR=1 \
python3 scripts/train_gciql.py \
        -cp $cp -cn $cn +cfg_path=$config \
        nclusters=40 \
        wandb_log=True \
        num_workers=0 batch_size=16 wandb_log=False \
        # augment_data=True \

#!/bin/bash
source ~/.bashrc
source activate stitch
alias python=python3

config="cfgs/gciql_luotest.yaml"

cp="../$(dirname $config)"
cn="$(basename $config)"
cn="${cn%.*}" # remove string after .
echo $cp $cn # echo $base_path

# HYDRA_FULL_ERROR=1 \
python3 scripts/train_gciql.py \
        -cp $cp -cn $cn +cfg_path=$config \
        augment_data=False \
        dataset_name=pointmaze-umaze-v0 nclusters=40 \
        # num_workers=0 batch_size=16 wandb_log=False \

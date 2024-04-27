#!/bin/bash
source ~/.bashrc
source activate stitch
alias python=python3
module load mesa

# config="cfgs/ben/ben_pointMmaze.yaml"

# config_list=( \
#     "cfgs/ben/ben_pointMmaze_aug05.yaml" \
# )


config_list=( \
    # "cfgs/ben/ben_antUmaze_aug05.yaml" \
    "cfgs/ben/ben_pointMmaze_aug05OnlyGoal.yaml" \
    "cfgs/ben/ben_pointMmaze_noaug.yaml" \
)

{
for config in "${config_list[@]}"; do
    cp="../$(dirname $config)"
    cn="$(basename $config)"
    cn="${cn%.*}" # remove string after .
    echo $cp $cn # echo $base_path
    # HYDRA_FULL_ERROR=1 \
    python3 scripts/train_dmlp_luo.py \
            -cp $cp -cn $cn +cfg_path=$config \
            # augment_data=False \
            # nclusters=40 \
            # wandb_log=False \
            # num_workers=0 batch_size=16 wandb_log=False \

done
exit 0
}
seed_list=(0 1 2 3 4)
# seed_list=(0)
# seed_list=(1 2 3 4)
env_list=("antmaze-umaze-v0" "antmaze-medium-v0" "antmaze-large-v0" "pointmaze-umaze-v0" "pointmaze-medium-v0" "pointmaze-large-v0")
# env_list=("antmaze-medium-v0")
is_augment="False"
augment_prob=0.0
num_updates_per_iter=40000
max_iters=25
batch_size=256
lr_list=(1e-3) # this is ignored in gciql

for lr in "${lr_list[@]}"
do
    for env_name in "${env_list[@]}"
    do
        for seed in "${seed_list[@]}"
        do
            if [ $env_name = "antmaze-umaze-v0" ]
            then
                sbatch scripts/template_gciql.sh -s ${seed} -d ${env_name} -a ${is_augment} -p ${augment_prob} -b ${batch_size} -l ${lr} -n 20 -u ${num_updates_per_iter} -i ${max_iters}
            fi

            if [ $env_name = "antmaze-medium-v0" ]
            then
                sbatch scripts/template_gciql.sh -s ${seed} -d ${env_name} -a ${is_augment} -p ${augment_prob} -b ${batch_size} -l ${lr} -n 40 -u ${num_updates_per_iter} -i ${max_iters}
            fi

            if [ $env_name = "antmaze-large-v0" ]
            then
                sbatch scripts/template_gciql.sh -s ${seed} -d ${env_name} -a ${is_augment} -p ${augment_prob} -b ${batch_size} -l ${lr} -n 80 -u ${num_updates_per_iter} -i ${max_iters}
            fi

            if [ $env_name = "pointmaze-umaze-v0" ]
            then
                sbatch scripts/template_gciql.sh -s ${seed} -d ${env_name} -a ${is_augment} -p ${augment_prob} -b ${batch_size} -l ${lr} -n 20 -u ${num_updates_per_iter} -i ${max_iters}
            fi

            if [ $env_name = "pointmaze-medium-v0" ]
            then
                sbatch scripts/template_gciql.sh -s ${seed} -d ${env_name} -a ${is_augment} -p ${augment_prob} -b ${batch_size} -l ${lr} -n 40 -u ${num_updates_per_iter} -i ${max_iters}
            fi

            if [ $env_name = "pointmaze-large-v0" ]
            then
                sbatch scripts/template_gciql.sh -s ${seed} -d ${env_name} -a ${is_augment} -p ${augment_prob} -b ${batch_size} -l ${lr} -n 80 -u ${num_updates_per_iter} -i ${max_iters}
            fi
        done
    done
done
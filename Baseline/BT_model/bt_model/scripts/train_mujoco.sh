#!/bin/bash
cd ..

#envs=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2"
#      "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
#envs=("halfcheetah-medium-expert-v2" "hopper-medium-expert-v2" "walker2d-medium-expert-v2")
#envs=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "hopper-medium-v2" "hopper-medium-replay-v2"
#      "walker2d-medium-v2" "walker2d-medium-replay-v2" )
#envs=("hopper-medium-v2" "hopper-medium-replay-v2" "walker2d-medium-v2")
#envs=("mo-hopper-expert_[0.1 0.1 0.8]")
envs=("cheetah-walk-rnd")

# crowdsourced labels (CS) + linear (MLP)
domain="mujoco"
modality="state"
structure="mlp"
fake_label=false
ensemble_size=3
n_epochs=200
num_query=2000
len_query=200
data_dir="../crowdsource_human_labels"
seed=999
exp_name="CS-MLP"

for env in "${envs[@]}"
do
        python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
        seed=$seed exp_name=$exp_name
done

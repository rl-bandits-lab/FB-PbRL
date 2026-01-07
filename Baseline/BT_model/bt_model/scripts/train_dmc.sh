#!/bin/bash
cd ..

envs=("cheetah-walk" "cheetah-walk_backward" "walker-walk")

# crowdsourced labels (CS) + linear (MLP)
domain="dmc"
modality="state"
structure="mlp"
fake_label=false
ensemble_size=3
n_epochs=200
num_query=2000
len_query=25
data_dir="../datasets/rnd_dmc_preference_dataset"
seed=0
exp_name="CS-MLP"

for env in "${envs[@]}"
do
        python train_reward_model.py domain=$domain env="$env" modality=$modality structure=$structure fake_label=$fake_label \
        ensemble_size=$ensemble_size n_epochs=$n_epochs num_query=$num_query len_query=$len_query data_dir=$data_dir \
        seed=$seed exp_name=$exp_name
done

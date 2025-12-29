#!/bin/bash
mkdir -p logs

SEED=${1:-0}

echo "Starting all runs with seed=$SEED" > logs/oprl_all.log

run() {
    echo "=== Running: $1 ==="
    eval "$1" >> logs/oprl_all.log 2>&1
    echo "=== Finished: $1 ==="
    echo "Sleeping 30s before next job..."
    sleep 30
}

run "python train_contrastive_reward.py --env walker-walk --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"
run "python train_contrastive_reward.py --env walker-stand --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"
run "python train_contrastive_reward.py --env walker-run --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"
run "python train_contrastive_reward.py --env walker-flip --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"

# cheetah
run "python train_contrastive_reward.py --env cheetah-walk --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"
run "python train_contrastive_reward.py --env cheetah-walk_backward --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"
run "python train_contrastive_reward.py --env cheetah-run --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"
run "python train_contrastive_reward.py --env cheetah-run_backward --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"

# quadruped
run "python train_contrastive_reward.py --env quadruped-walk --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"
run "python train_contrastive_reward.py --env quadruped-stand --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"
run "python train_contrastive_reward.py --env quadruped-run --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"
run "python train_contrastive_reward.py --env quadruped-jump --gpu 3 --teacher_eps_skip 0.05 --feed_type 'd' --seed $SEED"

echo "All experiments completed successfully with seed=$SEED." >> logs/oprl_all.log

# nohup ./oprl_reward_model.sh 1 > logs/oprl_run_all.out 2>&1 &
# Run all tasks in the background and log output to oprl_run_all.out & choose seed
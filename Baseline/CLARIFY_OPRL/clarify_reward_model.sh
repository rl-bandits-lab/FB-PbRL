#!/bin/bash
mkdir -p logs

: > logs/clarify_all.log

SEED=${1:-0}

echo "Starting all runs with seed=$SEED" > logs/clarify_all.log

run() {
    echo "=== Running: $1 ==="
    eval "$1" >> logs/clarify_all.log 2>&1
    echo "=== Finished: $1 ==="
    echo "Sleeping 30s before next job..."
    sleep 30
}

# walker
run "python train_contrastive_reward.py --env walker-walk --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"
run "python train_contrastive_reward.py --env walker-stand --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"
run "python train_contrastive_reward.py --env walker-run --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"
run "python train_contrastive_reward.py --env walker-flip --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"

# cheetah
run "python train_contrastive_reward.py --env cheetah-walk --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"
run "python train_contrastive_reward.py --env cheetah-walk_backward --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"
run "python train_contrastive_reward.py --env cheetah-run --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"
run "python train_contrastive_reward.py --env cheetah-run_backward --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"

# quadruped
run "python train_contrastive_reward.py --env quadruped-walk --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"
run "python train_contrastive_reward.py --env quadruped-stand --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"
run "python train_contrastive_reward.py --env quadruped-run --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"
run "python train_contrastive_reward.py --env quadruped-jump --gpu 2 --teacher_eps_skip 0.05 --feed_type 'c' --seed $SEED"

echo "All experiments completed successfully with seed=$SEED." >> logs/clarify_all.log

# nohup ./clarify_reward_model.sh 1 > logs/clarify_run_all.out 2>&1 &
# Run all tasks in the background and log output to clarify_run_all.out & choose seed
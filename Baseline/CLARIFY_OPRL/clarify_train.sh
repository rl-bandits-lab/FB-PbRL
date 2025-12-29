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
run "python policy_learning/oprl_policy.py --env walker-walk --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-walker-walk-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251225-145657" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env walker-stand --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-walker-stand-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251225-184928" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env walker-run --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-walker-run-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251225-224020" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env walker-flip --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-walker-flip-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251226-023130" --reward_model_name_mapping "scripts/reward_model_map_q50.json""

# cheetah
run "python policy_learning/oprl_policy.py --env cheetah-walk --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-cheetah-walk-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251226-062411" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env cheetah-walk_backward --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-cheetah-walk_backward-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251226-101715" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env cheetah-run --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-cheetah-run-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251226-140726" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env cheetah-run_backward --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-cheetah-run_backward-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251226-175713" --reward_model_name_mapping "scripts/reward_model_map_q50.json""

# quadruped
run "python policy_learning/oprl_policy.py --env quadruped-walk --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-quadruped-walk-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251226-214739" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env quadruped-stand --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-quadruped-stand-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251227-015054" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env quadruped-run --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-quadruped-run-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251227-055112" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env quadruped-jump --gpu 2 --teacher_eps_skip 0.05 --feed_type "c" --seed $SEED --reward_model_name "reward-quadruped-jump-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_c-ctx_200-seed_0-20251227-095334" --reward_model_name_mapping "scripts/reward_model_map_q50.json""

echo "All experiments completed successfully with seed=$SEED." >> logs/clarify_all.log

# nohup ./clarify_train.sh 1 > logs/clarify_run_all.out 2>&1 &
# Run all tasks in the background and log output to clarify_run_all.out & choose seed
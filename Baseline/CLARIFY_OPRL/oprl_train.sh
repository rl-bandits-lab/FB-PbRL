#!/bin/bash
mkdir -p logs

: > logs/oprl_all.log

SEED=${1:-0}

echo "Starting all runs with seed=$SEED" > logs/oprl_all.log

run() {
    echo "=== Running: $1 ==="
    eval "$1" >> logs/oprl_all.log 2>&1
    echo "=== Finished: $1 ==="
    echo "Sleeping 30s before next job..."
    sleep 30
}

# walker
run "python policy_learning/oprl_policy.py --env walker-walk --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-walker-walk-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251225-145725" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env walker-stand --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-walker-stand-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251225-185245" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env walker-run --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-walker-run-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251225-224257" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env walker-flip --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-walker-flip-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251226-023659" --reward_model_name_mapping "scripts/reward_model_map_q50.json""

# cheetah
run "python policy_learning/oprl_policy.py --env cheetah-walk --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-cheetah-walk-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251226-063056" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env cheetah-walk_backward --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-cheetah-walk_backward-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251226-102514" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env cheetah-run --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-cheetah-run-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251226-141459" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env cheetah-run_backward --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-cheetah-run_backward-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251226-180424" --reward_model_name_mapping "scripts/reward_model_map_q50.json""

# quadruped
run "python policy_learning/oprl_policy.py --env quadruped-walk --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-quadruped-walk-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251226-215737" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env quadruped-stand --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-quadruped-stand-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251227-015953" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env quadruped-run --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-quadruped-run-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251227-055834" --reward_model_name_mapping "scripts/reward_model_map_q50.json""
run "python policy_learning/oprl_policy.py --env quadruped-jump --gpu 3 --teacher_eps_skip 0.05 --feed_type "d" --seed $SEED --reward_model_name "reward-quadruped-jump-rnd-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_2000_q_50_skip_0.05_d-ctx_200-seed_0-20251227-100208" --reward_model_name_mapping "scripts/reward_model_map_q50.json""

echo "All experiments completed successfully." >> logs/oprl_all.log

# nohup ./oprl_train.sh 1 > logs/oprl_run_all.out 2>&1 &
# Run all tasks in the background and log output to opl_run_all.out & choose seed
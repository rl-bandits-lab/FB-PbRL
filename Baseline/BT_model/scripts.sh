python fb_train_dmc_bt.py --dataset_root datasets --domain_name cheetah --task_name walk --eval_tasks walk --device cuda --use_wandb
python fb_train_dmc_bt.py --dataset_root datasets --domain_name cheetah --task_name walk_backward --eval_tasks walk_backward --device cuda --use_wandb

python fb_test_dmc_bt.py --checkpoint tmp_fbcpr/20250923-000519-dmc-rnd-walker-walk/checkpoint --domain_name walker --task_name walk --device cuda
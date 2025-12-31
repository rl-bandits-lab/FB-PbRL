
---

## Pretraining

Run FB pretraining on DMC environments using offline RND datasets.

### Example: Cheetah
```bash
python fb_train_dmc.py \
  --domain_name cheetah \
  --dataset_root ./datasets_dmc \
  --dataset_type rnd \
  --use_wandb
```
## Fine-tuning

```bash
python fb_finetune_dmc_contrastive_hilp_dmc.py \
  --domain_name walker \
  --task_name run \
  --dataset_type rnd \
  --dataset_path ./datasets_dmc \
  --load_dir ./tmp_fbcpr/8N7POP9D1B/checkpoint \
  --num_train_steps 1000000 \
  --eval_every_steps 10000 \
  --device cuda \
  --use_contrastive \
  --use_dynamic_contrastive_z \
  --use_wandb \
  --contrastive_coef 100.0 \
```

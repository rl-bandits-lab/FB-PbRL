
---

## Dataset Sources

DeepMind Control Suite (DMC) offline datasets are obtained from the ExORL RND exploration benchmark. Download and save it in datasets_dmc folder

MetaWorld offline and preference datasets are adopted from LiRE.

Adroit Pen offline and preference datasets are taken from Preference Transformer.

## Pretraining
### DMC environment
Run FB pretraining on DMC environments using offline RND datasets.

```bash
python fb_train_dmc.py \
  --domain_name walker \
  --dataset_root ./datasets_dmc \
  --dataset_type rnd \
  --use_wandb
```

### Metaworld environment
```bash
python fb_train_metaworld.py \
  --domain_name metaworld \
  --task_name button-press-topdown-v2 \
  --env metaworld_button-press-topdown-v2 \
  --use_wandb \
  --human
```
## Fine-tuning

### Offline PbRL protocol

```bash
python fb_finetune_dmc_contrastive_hilp_dmc.py \
  --domain_name walker \
  --task_name walk \
  --dataset_type rnd \
  --dataset_path ./datasets_dmc \
  --load_dir pretrained_model/checkpoint \
  --num_train_steps 1000000 \
  --eval_every_steps 10000 \
  --device cuda \
  --use_contrastive \
  --use_dynamic_contrastive_z \
  --use_wandb \
  --contrastive_coef 100.0 \
```
The argument --num_pairs controls the size of the offline preference dataset.
The argument --noise specifies the preference noise level.

Offline preference datasets are collected using new_collect.py, which samples trajectory segments from the offline replay buffer and generates pairwise preferences using a scripted teacher. We collect 5,000 episodes for most DMC domains and 10,000 episodes for the PointMass domain. The teacher skip probability is set to teacher_eps_skip = 0.05 for all domains except PointMass, where it is set to 0.0.

### Zero-shot RL protocol

```bash
python fb_finetune_dmc_contrastive_hilp_dmc_zero_shot.py \
    --domain_name walker --task_name walk \
    --dataset_type rnd --dataset_path ./datasets_dmc \
    --load_dir pretrained_model/checkpoint \
    --num_train_steps 1000000 --eval_every_steps 10000 \
    --device cuda --use_contrastive --use_dynamic_contrastive_z \
    --use_wandb --contrastive_coef 100.0 --seq_length 25 --num_pairs 2000 \
```

Zero-shot preference datasets are collected using new_collect_zeroshot.py. The sequence length of each trajectory segment is controlled by --seq_length, and the number of preference pairs is specified by --num_pairs.

### Metaworld fine-tuning

```bash
python fb_finetune_metaword_contrastive.py \
  --env button-press-topdown-v2 \
  --num_pref_pairs 200 \
  --checkpoint pretrained_model/checkpoint \
  --num_train_steps 1000000 \
  --eval_every_steps 10000 \
  --device cuda \
  --use_contrastive \
  --use_dynamic_contrastive_z \
  --contrastive_coef 100.0 \
  --human \
  --segment_size 25 \
  --use_wandb
```

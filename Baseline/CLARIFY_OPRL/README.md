# CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries


This is the official implementation of CLARIFY (ICML 2025 poster, [arxiv](https://arxiv.org/abs/2506.00388), [OpenReview](https://openreview.net/forum?id=vOCPctm3nb)).

## Run experiments

First, you should make sure the 'logs' directory exists.

Train the reward model using CLARIFY, check clarify_reward_model.sh. For example:

```bash
nohup ./clarify_reward_model.sh 1 > logs/clarify_run_all.out 2>&1 &
```

Train the reward model using OPRL, check oprl_reward_model.sh. For example:

```bash
nohup ./oprl_reward_model.sh 1 > logs/oprl_run_all.out 2>&1 &
```

Train the offline policy based on CLARIFY's reward model, check clarify_train.sh. For example:

```bash
python scripts/reward_model_mapping.py
nohup ./clarify_train.sh 1 > logs/clarify_run_all.out 2>&1 &
```

Train the offline policy based on OPRL's reward model, check oprl_train.sh. For example:

```bash
python scripts/reward_model_mapping.py
nohup ./oprl_train.sh 1 > logs/oprl_run_all.out 2>&1 &
```



## Acknowledgement

This repo benefits from [LiRE](https://github.com/chwoong/LiRE), [HIM](https://github.com/frt03/generalized_dt) and [BPref](https://github.com/rll-research/BPref). Thanks for their wonderful work.


## Citation

If you find this project helpful, please consider citing the following paper:

```bibtex
@inproceedings{mu2025clarify,
    title={CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries},
    author={Mu, Ni and Hu, Hao and Hu, Xiao and Yang, Yiqin and XU, Bo and Jia, Qing-Shan},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025}
}
```



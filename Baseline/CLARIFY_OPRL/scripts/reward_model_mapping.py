import os
import re
import json

# results_dir = './results'
results_dir = './results/'
ablation_dir = None #'reject_sampling'
saved_reward_batch = 50
saved_max_feedback = 500
saved_seed_list = [3, 4, 5, 6, 7]

# env_values = ['button-press-topdown', 'drawer-open', 'box-close', 'dial-turn', \
#               'sweep', 'button-press-topdown-wall', 'sweep-into', 'lever-pull', \
#               'handle-pull-side', 'peg-insert-side', 'peg-unplug-side', 'hammer']
env_values = ['dial-turn', 'sweep-into']
env_values = ['cheetah-run', 'hopper-hop', 'walker-walk', 'humanoid-walk', 'quadruped-run', 'quadruped-walk', 'quadruped-stand', 'quadruped-jump', 'walker-stand', 'cheetah-run_backward', 'walker-run', 'cheetah-walk', 'cheetah-walk_backward', 'walker-flip']

pattern = r"reward-(?P<env>\S+-rnd).*_(?P<max_feedback>fb_(\d+)?)_(?P<reward_batch>q_(\d+)?)_(?P<skip>skip_\d(\.\d+)?)_(?P<feed_type>\S+)-ctx.*-seed_(?P<seed>\d+)-(?P<time>\d{8}-\d{6})"
# reward-dial-turn-medium-expert-conbdt-norm-0.1-comp-0.1-pref-1.0_fb_1000_q_50_skip_0.5_c-ctx_50-seed_1-20241230-015324
result_dict = {}

if __name__ == '__main__':
    list_dir = results_dir if ablation_dir == None else os.path.join(results_dir, ablation_dir)
    for subfolder in os.listdir(list_dir):
        # if ablation_dir != None:
        #     subfolder_path = os.path.join(results_dir, ablation_dir, subfolder)
        # else:
        subfolder_path = os.path.join(list_dir, subfolder)

        if os.path.isdir(subfolder_path):
            print(f"Checking {subfolder}")
            # 使用正则表达式匹配文件夹名
            match = re.match(pattern, subfolder)

            if match:
                print(f"Match!")
                # 提取匹配到的值
                env = match.group('env')[:-4]
                if env not in env_values:
                    print(f"Env {env} not in list")
                    continue
                reward_batch = int(match.group('reward_batch').split('_')[1])
                if reward_batch != saved_reward_batch:
                    continue
                max_feedback = int(match.group('max_feedback').split('_')[1])
                # if max_feedback != saved_max_feedback:
                #     continue
                skip = float(match.group('skip').split('_')[1])  # 转换为 float
                feed_type = match.group('feed_type')
                seed = int(match.group('seed'))
                # if not seed in saved_seed_list:
                #     continue 
                timestamp = match.group('time')

                # 生成 key: env_skip_seed 格式
                key = f"{env}_{skip}_{feed_type}_{seed}"
                print(f"Key: {key}")

                # 如果该 key 没有记录或者找到更晚的时间戳，更新记录
                if key not in result_dict or result_dict[key]['time'] < timestamp:
                    result_dict[key] = {
                        'folder': f'{ablation_dir}/{subfolder}' if ablation_dir != None else subfolder,
                        'time': timestamp
                    }
                    print(f"Updated {key} to {subfolder}")

    save_dict = {k: v['folder'] for k, v in result_dict.items()}
    if ablation_dir != None:
        json_output_path = f'./scripts/reward_model_map_q{saved_reward_batch}_{ablation_dir}.json'
    else:
        json_output_path = f'./scripts/reward_model_map_q{saved_reward_batch}.json'
    with open(json_output_path, 'w') as json_file:
        json.dump(save_dict, json_file, indent=4)

    print(f"Matching folders saved to {json_output_path}")



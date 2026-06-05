#!/usr/bin/env python
"""
Train Base PPO

便捷脚本，用于训练基础的 PPO（速度控制）。
训练完成后保存模型，用于 Preference Stage。

使用方法：
    # 基础训练
    python scripts/train_base_ppo.py --config_template real_data

    # 带 WandB 追踪
    python scripts/train_base_ppo.py --config_template real_data --track

    # 自定义参数
    python scripts/train_base_ppo.py \
        --config_template real_data \
        --total_timesteps 1000000 \
        --num_uavs 1 \
        --seed 42

训练完成后，模型保存在 models/ 目录。
使用 Preference Stage 时，指定 --base_ppo_path 参数。
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入并运行训练
from agents.RPO_ContinusActionSpace import Args
import tyro

if __name__ == "__main__":
    # 解析参数并运行
    # 注意：直接调用 agents/RPO_ContinusActionSpace.py 的 main
    import subprocess
    
    # 传递所有命令行参数，并把项目根加入 PYTHONPATH 以便子进程能找到 environments 等包
    cmd = [sys.executable, os.path.join(project_root, 'agents', 'RPO_ContinusActionSpace.py')] + sys.argv[1:]
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    
    print(f"[train_base_ppo] Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)

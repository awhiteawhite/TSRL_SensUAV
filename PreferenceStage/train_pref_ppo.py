"""
Train Preference Stage PPO

训练 Preference Stage 的 PPO 策略。

系统架构：
- Preference Policy: 选择 (UAV, sensing point) 配对
- Frozen PPO: 执行 K 步速度决策
- 奖励 = order_weight * 订单奖励 + sensing_weight * 感知奖励（order_weight 由下层 UAVEnvironment.reward_config 控制）

使用方法：
    python PreferenceStage/train_pref_ppo.py \
        --base_ppo_path models/base_ppo/best_model.pt \
        --config_template real_data \
        --total_timesteps 500000
"""

import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pref_policy import PreferenceActorCritic, masked_softmax
from baseline.DECO_baseline import PreferenceActorCritic as DECOActorCritic
from single_stage_policy import SingleStagePairPolicy 
from baseline.D2SN_baseline import D2SNSingleStepPairPolicy
from baseline.DyPS_baseline import DyPSHierarchicalPolicy
from baseline.SMORE_baseline import SMOREBaseline
from pref_buffer import PreferenceRolloutBuffery
from pref_utils import (
    sample_uav_action, sample_sens_action, sample_pair_action,
    obs_dict_to_tensor, compute_ppo_loss, compute_value_loss,
    explained_variance, generate_sensing_points
)
from macro_env import MacroPreferenceEnv, create_macro_env

from environments.uav_environment import UAVEnvironment
from agents.load_ppo_policy import load_frozen_policy
from config import get_config_template
from datasets.dataset_manager import DatasetManager


@dataclass
class Args:
    """训练配置"""
    # 基础配置
    config_template: str = "real_data"
    """环境配置模板"""
    exp_name: str = "PreferencePPO"
    """实验名称"""
    seed: int = 1
    """随机种子"""
    torch_deterministic: bool = True
    """PyTorch 确定性"""
    cuda: bool = True
    """使用 CUDA"""
    track: bool = False
    """使用 WandB"""
    wandb_project_name: str = "UAV_PreferencePPO"
    """WandB 项目名"""
    
    # 模型路径
    base_ppo_path: Optional[str] = None
    """预训练的 base PPO 模型路径"""
    
    # 环境配置
    num_uavs: int = 10
    """UAV 数量"""
    num_sensing_points: int = 20
    """感知点数量"""
    K: int = 6
    """每个 macro step 的 low-level step 数量"""
    sensing_weight: float = 1.0
    """感知超时惩罚宏观放大系数（订单权重 order_weight 由 UAVEnvironment.reward_config 统一管理）"""
    
    # 训练配置
    total_timesteps: int = 500000
    """总 macro step 数"""
    learning_rate: float = 3e-4
    """学习率"""
    num_steps: int = 256
    """每次 rollout 的 macro step 数"""
    num_minibatches: int = 4
    """minibatch 数量"""
    update_epochs: int = 4
    """每次更新的 epoch 数"""
    
    # PPO 参数
    gamma: float = 0.99
    """折扣因子"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    clip_coef: float = 0.2
    """PPO clip 系数"""
    clip_vloss: bool = True
    """是否 clip 价值损失"""
    ent_coef: float = 0.01
    """熵系数"""
    vf_coef: float = 0.5
    """价值函数系数"""
    max_grad_norm: float = 0.5
    """梯度裁剪"""
    normalize_advantage: bool = True
    """是否归一化优势"""
    
    # 网络配置
    d_emb: int = 128
    """嵌入维度"""
    d_hidden: int = 256
    """隐藏层维度"""
    policy_variant: str = "factorized"
    """策略架构: factorized | single_stage_pair | d2sn | dyps | smore"""
    dyps_cvae_coef: float = 0.0
    """dyps CVAE 辅助损失权重（>0 启用 ELBO 监督；0=不加 CVAE 损失）"""
    low_level_mode: str = "frozen_ppo"
    """低层速度决策: frozen_ppo | fixed_speed"""
    fixed_speed_value: float = 0.5
    """fixed_speed 模式下的速度比例 [0,1]，实际速度 = value * max_speed"""
    
    # 日志配置
    log_interval: int = 10
    """日志打印间隔"""
    save_interval: int = 100
    """模型保存间隔（update 次数）"""
    model_save_dir: str = "models/preference_ppo"
    """模型保存目录"""
    
    # 数据集划分
    split_path: str = "datasets/data_split.json"
    """数据集划分 JSON 路径"""
    eval_interval: int = 50
    """验证集评估间隔（update 次数，0=不评估）"""
    max_eval_episodes: int = 10
    """验证时最多评估的 episode 数"""
    
    # 评估模式
    eval_only: bool = False
    """仅评估，不训练（需要 --eval_model_path）"""
    eval_model_path: Optional[str] = None
    """评估用的模型路径（.pt 文件）"""
    eval_split: str = "test"
    """评估使用的数据子集: train / val / test"""
    path_log_dir: Optional[str] = None
    """非空时在 eval_only/evaluate_on_dates 中保存 UAV 路径 CSV，用于可视化平台 API 对齐"""
    path_log_prefix: Optional[str] = None
    """path_log CSV 文件名前缀（默认 = policy_variant）；用于在 PSPP / DECO 共用 factorized variant 时区分输出文件"""

    # 超参数实验 - reward 引导系数（None=使用 JSON config 的值，CLI 传值时直接覆盖 config）
    order_timeout_penalty: Optional[float] = None
    """订单超时惩罚（如 -500）；None=使用 config"""
    sensing_timeout_penalty: Optional[float] = None
    """感知超时惩罚 raw 值（如 -5），实际有效值还会乘 sensing_weight；None=使用 config"""
    shaping_reward_coeff: Optional[float] = None
    """shaping reward 系数（如 0.0002）；None=使用 config"""
    max_speed: Optional[float] = None
    """最大速度（如 500）；None=使用 config"""


def make_env(args: Args, env_config: Dict, dataset_manager: DatasetManager) -> MacroPreferenceEnv:
    """
    创建 MacroPreferenceEnv
    
    注意：
    - args 中的参数已在 train() 函数中被 JSON 配置覆盖
    - env_config 是展平后的配置字典（所有参数在顶层）
    """
    # 创建 base 环境
    region_id = env_config.get('region_id', 0)
    base_env = UAVEnvironment(
        config=env_config,
        num_uavs=args.num_uavs,  # 已从 JSON 读取
        dataset_manager=dataset_manager,
        episode_sampling_mode='daily',
        region_id=region_id,
    )
    
    # 加载低层速度决策器
    _device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.low_level_mode == "fixed_speed":
        low_level_policy = None
        print(f"[make_env] low_level_mode=fixed_speed, value={args.fixed_speed_value}")
    else:
        low_level_policy = load_frozen_policy(
            model_path=args.base_ppo_path,
            num_uavs=args.num_uavs,
            device=_device,
        )
    
    # 从展平的 env_config 读取时间窗参数
    time_window_min = env_config.get('time_window_minutes_min', 600)
    time_window_max = env_config.get('time_window_minutes_max', 900)
    priority_min = env_config.get('priority_range_min', 0.5)
    priority_max = env_config.get('priority_range_max', 1.0)
    
    # 生成感知点（基于环境的区域边界）
    region_bounds = base_env.region_bounds
    sensing_points = generate_sensing_points(
        region_bounds=region_bounds,
        num_points=args.num_sensing_points,
        time_window_minutes=(time_window_min, time_window_max),
        priority_range=(priority_min, priority_max),
        seed=args.seed
    )
    print(f"[make_env] Generated {len(sensing_points)} sensing points (seed=42 fixed) with time_window=({time_window_min}, {time_window_max}) min")
    
    # reset env 使得 current_time 有效
    base_env.reset(seed=args.seed)
    
    # 创建 macro 环境
    macro_env = create_macro_env(
        base_env=base_env,
        low_level_policy=low_level_policy,
        sensing_points=sensing_points,
        K=args.K,
        sensing_weight=args.sensing_weight,
        max_sensing_points=args.num_sensing_points + 10,
        sensing_timeout_penalty=env_config.get('sensing_timeout_penalty', -5.0),
        sensing_completion_distance=env_config.get('sensing_completion_distance', 100.0),
        fixed_speed_value=args.fixed_speed_value,
    )
    
    return macro_env


def _smore_stage2_kwargs(policy, obs, uav_action):
    """SMORE-only: 透传 raw uav_feats / sens_feats / chosen_uav_idx 给 stage-2，
    让 SMORE 内部基于物理位置 + priority/data_value 计算 delta_cov / delta_inc。
    对其他 baseline 返回空 dict，调用点变成 no-op，**不影响 D2SN / DyPS / PreferenceActorCritic**。

    Args:
        policy: 当前 policy 模块
        obs: dict（rollout / eval 路径）或 buffer batch namedtuple（PPO update 路径）
        uav_action: (B,) 张量或标量张量，stage-1 选定的 UAV idx
    """
    if not isinstance(policy, SMOREBaseline):
        return {}
    if isinstance(obs, dict):
        uav_feats = obs['uav_feats']
        sens_feats = obs['sens_feats']
    else:
        uav_feats = obs.uav_feats
        sens_feats = obs.sens_feats
    chosen_idx = uav_action.detach() if torch.is_tensor(uav_action) else uav_action
    return {
        'uav_feats': uav_feats,
        'sens_feats': sens_feats,
        'chosen_uav_idx': chosen_idx,
    }


def _eval_step_factorized(policy, obs_t):
    """Factorized: sample UAV then conditional sensing (deterministic)."""
    uav_logits, _, cache = policy(obs_t)
    uav_action, _, _, _ = sample_uav_action(uav_logits, obs_t['uav_mask'], deterministic=True)
    uav_e = cache['uav_e']
    B = uav_e.size(0)
    idx = uav_action.view(B, 1, 1).expand(B, 1, uav_e.size(-1))
    chosen_uav_e = uav_e.gather(1, idx).squeeze(1)
    smore_kw = _smore_stage2_kwargs(policy, obs_t, uav_action)
    sens_logits, sens_mask_all = policy.conditional_sens_logits(
        chosen_uav_e, cache['sens_e'], obs_t['sens_mask'], obs_t['global_feats'],
        **smore_kw,
    )
    sens_action, _, _, _ = sample_sens_action(sens_logits, sens_mask_all, deterministic=True)
    return int(uav_action.item()), int(sens_action.item())


def _eval_step_single_stage(policy, obs_t):
    """Single-stage: sample pair then decode (deterministic)."""
    pair_logits, pair_mask, _, _ = policy(obs_t)
    pair_action, _, _, _ = sample_pair_action(pair_logits, pair_mask, deterministic=True)
    M1 = obs_t['sens_feats'].size(1) + 1
    uav_id, sens_id = SingleStagePairPolicy.decode_pair_action(pair_action.item(), M1)
    return int(uav_id), int(sens_id)


@torch.no_grad()
def evaluate_on_dates(
    policy,
    macro_env: MacroPreferenceEnv,
    dates: List[str],
    region_bounds,
    env_config: Dict,
    args: Args,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, float]:
    """在给定日期列表上评估 policy（验证/测试），返回汇总指标。"""
    policy.eval()
    is_single_stage = args.policy_variant in ("single_stage_pair", "d2sn", "dyps")

    num_sensing_points = args.num_sensing_points
    tw_min = env_config.get('time_window_minutes_min', 600)
    tw_max = env_config.get('time_window_minutes_max', 900)
    pri_min = env_config.get('priority_range_min', 0.5)
    pri_max = env_config.get('priority_range_max', 1.0)

    all_rewards, all_orders_comp, all_orders_total = [], [], []
    all_sens_comp, all_sens_total = [], []
    # --- 业务 objective 分量（干净，不含 timeout / shaping / 训练系数）---
    all_sensing_entropy, all_delivery_profit, all_energy_cost = [], [], []
    # --- 辅助：timeout / shaping 拆开单独记录 ---
    all_sensing_timeout, all_order_timeout, all_shaping = [], [], []
    if args.path_log_dir:
        os.makedirs(args.path_log_dir, exist_ok=True)

    for ep_idx, date_str in enumerate(dates):
        ep_seed = seed + ep_idx
        sensing_points = generate_sensing_points(
            region_bounds=region_bounds,
            num_points=num_sensing_points,
            time_window_minutes=(tw_min, tw_max),
            priority_range=(pri_min, pri_max),
            seed=ep_seed,
        )
        macro_env.base_env.allowed_dates = [date_str]
        obs = macro_env.reset(seed=ep_seed, sensing_points=sensing_points)
        if args.path_log_dir:
            from PreferenceStage.path_logger import (
                dump_sensing_tasks_csv,
                dump_delivery_orders_csv,
            )
            log_prefix = args.path_log_prefix if args.path_log_prefix else args.policy_variant
            sensing_log_path = dump_sensing_tasks_csv(
                out_dir=args.path_log_dir,
                prefix=log_prefix,
                split=args.eval_split,
                date_str=date_str,
                seed=seed,
                ep_idx=ep_idx,
                ep_seed=ep_seed,
                sensing_points=macro_env.pref_wrapper.sensing_points,
            )
            print(f"[Sensing Task Log] -> {sensing_log_path}")
            orders_log_path = dump_delivery_orders_csv(
                out_dir=args.path_log_dir,
                prefix=log_prefix,
                split=args.eval_split,
                date_str=date_str,
                seed=seed,
                ep_idx=ep_idx,
                ep_seed=ep_seed,
                base_env=macro_env.base_env,
            )
            print(f"[Delivery Orders Log] -> {orders_log_path}")

        ep_reward = 0.0
        ep_sensing_entropy = 0.0   # 纯 sensing 熵收益
        ep_delivery_profit = 0.0   # 纯配送完成收益
        ep_energy_cost = 0.0       # 能量消耗（正值）
        ep_sensing_timeout = 0.0   # sensing 超时惩罚（原始，未加权）
        ep_order_timeout = 0.0     # 订单超时惩罚（已乘 order_weight）
        ep_shaping = 0.0           # shaping 奖励
        done = False
        info = {}
        path_rows = []
        step_idx = 0
        while not done:
            obs_t = obs_dict_to_tensor(obs, device)
            if is_single_stage:
                uav_id, sens_id = _eval_step_single_stage(policy, obs_t)
            else:
                uav_id, sens_id = _eval_step_factorized(policy, obs_t)

            next_obs, reward, terminated, truncated, info = macro_env.step((uav_id, sens_id))
            if args.path_log_dir:
                from PreferenceStage.path_logger import append_step_rows
                append_step_rows(
                    buffer=path_rows,
                    info=info,
                    date_str=date_str,
                    ep_idx=ep_idx,
                    step_idx=step_idx,
                    uav_id_decision=uav_id,
                    sens_id_decision=sens_id,
                    base_env=macro_env.base_env,
                )
            ep_reward          += reward
            ep_sensing_entropy += info.get('sensing_entropy_reward', 0.0)
            ep_delivery_profit += info.get('delivery_completion_reward', 0.0)
            ep_energy_cost     += abs(info.get('energy_penalty', 0.0))
            ep_sensing_timeout += info.get('sensing_timeout_penalty', 0.0)
            ep_order_timeout   += info.get('order_timeout_penalty', 0.0)
            ep_shaping         += info.get('shaping_reward', 0.0)
            obs = next_obs
            done = terminated or truncated
            step_idx += 1

        all_rewards.append(ep_reward)
        all_sensing_entropy.append(ep_sensing_entropy)
        all_delivery_profit.append(ep_delivery_profit)
        all_energy_cost.append(ep_energy_cost)
        all_sensing_timeout.append(ep_sensing_timeout)
        all_order_timeout.append(ep_order_timeout)
        all_shaping.append(ep_shaping)
        sensing_stats = info.get('sensing_stats', {})
        all_orders_comp.append(info.get('orders_completed', 0))
        ot = info.get('total_orders_in_episode', 0)
        if ot == 0:
            be = macro_env.base_env
            if hasattr(be, 'processed_orders') and be.processed_orders:
                ot = len(be.processed_orders)
            elif getattr(be, 'current_episode', None) is not None:
                ot = len(be.current_episode.orders)
        all_orders_total.append(ot)
        all_sens_comp.append(sensing_stats.get('completed_sensing_points', 0))
        all_sens_total.append(sensing_stats.get('total_sensing_points', 0))
        if args.path_log_dir:
            from PreferenceStage.path_logger import dump_path_csv
            log_prefix = args.path_log_prefix if args.path_log_prefix else args.policy_variant
            path_log_path = dump_path_csv(
                out_dir=args.path_log_dir,
                prefix=log_prefix,
                split=args.eval_split,
                date_str=date_str,
                seed=seed,
                ep_idx=ep_idx,
                rows=path_rows,
            )
            print(f"[Path Log] saved {len(path_rows)} rows -> {path_log_path}")

    policy.train()

    avg_sensing_entropy  = float(np.mean(all_sensing_entropy))
    avg_delivery_profit  = float(np.mean(all_delivery_profit))
    avg_energy_cost      = float(np.mean(all_energy_cost))
    avg_energy_safe      = avg_energy_cost + 1e-8            # 避免除零

    so = sum(all_orders_total)
    ss = sum(all_sens_total)
    out = {
        # ---- 业务 objective（论文指标）----
        'avg_objective': avg_sensing_entropy + avg_delivery_profit - avg_energy_cost,
        'avg_sensing_entropy': avg_sensing_entropy,
        'avg_delivery_profit': avg_delivery_profit,
        'avg_energy_cost': avg_energy_cost,
        'sensing_entropy_per_energy': avg_sensing_entropy / avg_energy_safe,
        'delivery_profit_per_energy': avg_delivery_profit  / avg_energy_safe,
        # ---- 辅助：训练监控 ----
        'avg_train_reward': float(np.mean(all_rewards)),    # 训练 reward（含 shaping/timeout/系数）
        'avg_sensing_timeout': float(np.mean(all_sensing_timeout)),
        'avg_order_timeout': float(np.mean(all_order_timeout)),
        'avg_shaping': float(np.mean(all_shaping)),
        # ---- 完成率 ----
        'order_completion_rate': sum(all_orders_comp) / so if so > 0 else 0.0,
        'sensing_completion_rate': sum(all_sens_comp) / ss if ss > 0 else 0.0,
        'std_objective': float(np.std(
            [s + d - e for s, d, e in zip(all_sensing_entropy, all_delivery_profit, all_energy_cost)]
        )),
        'num_episodes': len(dates),
    }

    return out


def train(args: Args):
    """主训练函数"""
    # 加载配置（先加载，用于覆盖 args 默认值）
    # 注意：get_config_template 返回的是展平后的配置字典
    env_config = get_config_template(args.config_template)
    
    # ========== CLI 优先：仅当 CLI 未显式设置（仍为默认值）时才从 JSON 覆盖 ==========
    _defaults = Args()
    _override_keys = [
        'num_uavs', 'K', 'sensing_weight',
        'd_emb', 'd_hidden', 'num_sensing_points',
        'seed', 'learning_rate', 'num_steps', 'num_minibatches',
        'update_epochs', 'gamma', 'gae_lambda', 'clip_coef',
        'ent_coef', 'vf_coef', 'max_grad_norm',
        'exp_name', 'wandb_project_name',
        'log_interval', 'save_interval', 'model_save_dir',
        'eval_interval', 'max_eval_episodes',
        'policy_variant', 'low_level_mode', 'fixed_speed_value','max_speed'
    ]
    for _k in _override_keys:
        if _k in env_config and getattr(args, _k) == getattr(_defaults, _k):
            setattr(args, _k, env_config[_k])

    # ========== CLI → env_config 方向：超参数实验用，CLI 传值时覆盖 JSON ==========
    # 这三个参数后续从 env_config.get() 读取（不经过 args），需要写回 env_config 才能生效
    if args.order_timeout_penalty is not None:
        env_config['order_timeout_penalty'] = args.order_timeout_penalty
    if args.sensing_timeout_penalty is not None:
        env_config['sensing_timeout_penalty'] = args.sensing_timeout_penalty
    if args.shaping_reward_coeff is not None:
        env_config['shaping_reward_coeff'] = args.shaping_reward_coeff
    # ========== End CLI → env_config ==========

    print(f"[Config] Loaded from '{args.config_template}.json'")
    print(f"  num_uavs={args.num_uavs}, K={args.K}, num_sensing_points={args.num_sensing_points}")
    print(f"  sensing_weight={args.sensing_weight}")
    print(f"  reward formula: sensing + delivery + 2*energy + shaping")
    print(f"  d_emb={args.d_emb}, d_hidden={args.d_hidden}")
    print(f"  policy_variant={args.policy_variant}")
    print(f"  low_level_mode={args.low_level_mode}, fixed_speed_value={args.fixed_speed_value}")
    # ========== End 配置覆盖 ==========
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建运行名称
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # 初始化 WandB
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
        )
    
    # TensorBoard
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", str(vars(args)))
    
    # 创建 DatasetManager
    dataset_path = env_config.get('dataset_path', 'datasets')
    orders_filename = env_config.get('orders_filename', 'hangzhou_region0_101_MayToJuly.csv')
    csv_path = os.path.join(dataset_path, orders_filename)
    dataset_manager = DatasetManager(csv_path, config_template=args.config_template)
    
    # ========== 数据集划分：train / val / test ==========
    region_id = env_config.get('region_id', 0)
    split_path = env_config.get('split_path', args.split_path)
    split_mode = env_config.get('split_mode', 'random')
    split_seed = env_config.get('split_seed', 42)
    train_ratio = env_config.get('train_ratio', 0.78)
    val_ratio = env_config.get('val_ratio', 0.11)
    test_ratio = env_config.get('test_ratio', 0.11)
    
    train_dates, val_dates, test_dates = DatasetManager.get_split_dates(
        dataset_manager=dataset_manager,
        region_id=str(region_id),
        split_path=split_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        mode=split_mode,
        seed=split_seed,
    )
    print(f"[Data Split] train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)}")
    
    # 创建环境
    print("[Creating Environment]")
    env = make_env(args, env_config, dataset_manager)
    
    # 训练时只使用 train_dates
    env.base_env.allowed_dates = train_dates

    # 一次性将所有训练日期的 EpisodeData 加载到内存，训练期间 env.reset() 零 I/O
    _t0 = time.time()
    preloaded = dataset_manager.preload_episodes(
        dates=train_dates,
        region_id=str(region_id),
    )
    env.base_env.preloaded_episodes = preloaded
    print(f"[Preload] {len(preloaded)} training episodes cached in {time.time()-_t0:.1f}s")
    
    print(f"  Num UAVs: {env.num_uavs}")
    print(f"  Max Sensing Points: {env.max_sensing_points}")
    print(f"  K (low-level steps per macro step): {args.K}")
    print(f"  Training on {len(train_dates)} dates (out of {len(train_dates)+len(val_dates)+len(test_dates)} total)")
    
    # 创建 Preference Policy
    print(f"\n[Creating Preference Policy — variant={args.policy_variant}]")
    d_uav = 9
    d_sens = 8
    d_global = 4
    is_single_stage = args.policy_variant in ("single_stage_pair", "d2sn", "dyps")

    if args.policy_variant == "d2sn":
        policy = D2SNSingleStepPairPolicy(
            duav=d_uav,
            dsens=d_sens,
            dglobal=d_global,
            demb=args.d_emb,
            hid=args.d_hidden,
            max_uavs=env.num_uavs,
            max_sens=env.max_sensing_points,
        ).to(device)
    elif args.policy_variant == "dyps":
        policy = DyPSHierarchicalPolicy(
            duav=d_uav,
            dsens=d_sens,
            dglobal=d_global,
            dmodel=args.d_emb,
            hid=args.d_hidden,
            max_uavs=env.num_uavs,
            max_sens=env.max_sensing_points,
        ).to(device)
    elif args.policy_variant == "single_stage_pair":
        policy = SingleStagePairPolicy(
            duav=d_uav,
            dsens=d_sens,
            dglobal=d_global,
            demb=args.d_emb,
            hid=args.d_hidden,
            max_uavs=env.num_uavs,
            max_sens=env.max_sensing_points,
        ).to(device)
    elif args.policy_variant == "smore":
        policy = SMOREBaseline(
            duav=d_uav,
            dsens=d_sens,
            dglobal=d_global,
            demb=args.d_emb,
            hid=args.d_hidden,
            max_uavs=env.num_uavs,
            max_sens=env.max_sensing_points,
        ).to(device)
    else:
        policy = PreferenceActorCritic(
            duav=d_uav,
            dsens=d_sens,
            dglobal=d_global,
            demb=args.d_emb,
            hid=args.d_hidden,
            max_sens=env.max_sensing_points,
        ).to(device)
    print(f"  Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # 创建 Rollout Buffer
    buffer = PreferenceRolloutBuffer(
        buffer_size=args.num_steps,
        max_uavs=env.num_uavs,
        max_sensing_points=env.max_sensing_points,
        d_uav=d_uav,
        d_sens=d_sens,
        d_global=d_global,
        device=device,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    
    # 计算 batch 大小
    batch_size = args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    
    print(f"\n[Training Config]")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Minibatch size: {minibatch_size}")
    print(f"  Num updates: {num_updates}")
    
    # 重置环境
    obs = env.reset(seed=args.seed)
    
    global_step = 0
    start_time = time.time()
    episode_count = 0
    episode_return = 0.0
    # ---- 业务 objective 分量（干净，不含 timeout / shaping）----
    episode_sensing_entropy = 0.0
    episode_delivery_profit = 0.0
    episode_energy_cost = 0.0
    # ---- 辅助：timeout / shaping 单独记录 ----
    episode_sensing_timeout = 0.0
    episode_order_timeout = 0.0
    episode_shaping = 0.0
    best_val_reward = -float('inf')
    region_bounds_for_eval = env.base_env.region_bounds
    
    print("\n[Training Start]")
    print("=" * 60)
    
    for update in range(1, num_updates + 1):
        # ========== Rollout Collection ==========
        buffer.reset()
        
        for step in range(args.num_steps):
            global_step += 1
            
            obs_tensor = obs_dict_to_tensor(obs, device)
            
            with torch.no_grad():
                if is_single_stage:
                    # ---------- single-stage pair sampling ----------
                    pair_logits, pair_mask, value, _ = policy(obs_tensor)
                    pair_action, pair_lp, _, _ = sample_pair_action(
                        pair_logits, pair_mask, deterministic=False
                    )
                    M1 = obs_tensor['sens_feats'].size(1) + 1
                    uav_action_t, sens_action_t = SingleStagePairPolicy.decode_pair_action(pair_action, M1)
                    uav_id = int(uav_action_t.item())
                    sens_id = int(sens_action_t.item())
                    uav_log_prob = pair_lp       # store pair lp as uav_log_prob
                    sens_log_prob = torch.zeros_like(pair_lp)
                else:
                    # ---------- factorized two-stage sampling ----------
                    uav_logits, value, cache = policy(obs_tensor)
                    uav_action, uav_log_prob, _, _ = sample_uav_action(
                        uav_logits, obs_tensor['uav_mask'], deterministic=False
                    )
                    uav_e = cache['uav_e']
                    sens_e = cache['sens_e']
                    B = uav_e.size(0)
                    idx = uav_action.view(B, 1, 1).expand(B, 1, uav_e.size(-1))
                    chosen_uav_e = uav_e.gather(1, idx).squeeze(1)
                    smore_kw = _smore_stage2_kwargs(policy, obs_tensor, uav_action)
                    sens_logits, sens_mask_all = policy.conditional_sens_logits(
                        chosen_uav_e, sens_e, obs_tensor['sens_mask'], obs_tensor['global_feats'],
                        **smore_kw,
                    )
                    sens_action, sens_log_prob, _, _ = sample_sens_action(
                        sens_logits, sens_mask_all, deterministic=False
                    )
                    uav_id = int(uav_action.item())
                    sens_id = int(sens_action.item())
            
            # 环境 step
            next_obs, reward, terminated, truncated, info = env.step((uav_id, sens_id))
            done = terminated or truncated
            
            # 存储到 buffer
            buffer.add(
                obs=obs,
                uav_action=uav_id,
                sens_action=sens_id,
                reward=reward,
                done=done,
                value=value.item(),
                uav_log_prob=uav_log_prob.item(),
                sens_log_prob=sens_log_prob.item(),
            )
            
            # 更新观察
            obs = next_obs
            
            # 累积 episode 各分量
            episode_return          += reward
            episode_sensing_entropy += info.get('sensing_entropy_reward', 0.0)
            episode_delivery_profit += info.get('delivery_completion_reward', 0.0)
            episode_energy_cost     += abs(info.get('energy_penalty', 0.0))
            episode_sensing_timeout += info.get('sensing_timeout_penalty', 0.0)
            episode_order_timeout   += info.get('order_timeout_penalty', 0.0)
            episode_shaping         += info.get('shaping_reward', 0.0)

            # Episode 结束处理
            if done:
                episode_count += 1
                episode_objective = episode_sensing_entropy + episode_delivery_profit - episode_energy_cost
                _e_safe = episode_energy_cost + 1e-8

                # 打印 episode 信息（业务 objective 口径）
                sensing_stats = info.get('sensing_stats', {})
                print(f"Episode {episode_count}: "
                      f"macro_steps={info.get('macro_step', 0)}, "
                      f"sensing={sensing_stats.get('completed_sensing_points', 0)}/{sensing_stats.get('total_sensing_points', 0)}, "
                      f"orders={info.get('orders_completed', 0)}, "
                      f"objective={episode_objective:.2f} "
                      f"(sensing_entropy={episode_sensing_entropy:.2f}, "
                      f"delivery_profit={episode_delivery_profit:.2f}, "
                      f"energy_cost={episode_energy_cost:.4f})")

                # TensorBoard / WandB 记录
                # ---- 业务 objective（论文指标）----
                writer.add_scalar("train/objective",                  episode_objective,                        global_step)
                writer.add_scalar("train/sensing_entropy",            episode_sensing_entropy,                  global_step)
                writer.add_scalar("train/delivery_profit",            episode_delivery_profit,                  global_step)
                writer.add_scalar("train/energy_cost",                episode_energy_cost,                      global_step)
                writer.add_scalar("train/sensing_entropy_per_energy", episode_sensing_entropy / _e_safe,        global_step)
                writer.add_scalar("train/delivery_profit_per_energy", episode_delivery_profit / _e_safe,        global_step)
                # ---- 辅助：训练监控 ----
                writer.add_scalar("train/episodic_return",            episode_return,                           global_step)
                writer.add_scalar("train/sensing_timeout",            episode_sensing_timeout,                  global_step)
                writer.add_scalar("train/order_timeout",              episode_order_timeout,                    global_step)
                writer.add_scalar("train/shaping_reward",             episode_shaping,                          global_step)
                # ---- 完成情况 ----
                writer.add_scalar("episode/macro_steps",              info.get('macro_step', 0),                global_step)
                writer.add_scalar("episode/sensing_completed",        sensing_stats.get('completed_sensing_points', 0), global_step)
                writer.add_scalar("episode/orders_completed",         info.get('orders_completed', 0),          global_step)

                # 重置 episode 统计
                episode_return          = 0.0
                episode_sensing_entropy = 0.0
                episode_delivery_profit = 0.0
                episode_energy_cost     = 0.0
                episode_sensing_timeout = 0.0
                episode_order_timeout   = 0.0
                episode_shaping         = 0.0
                
                # 重置环境
                obs = env.reset()
        
        # ========== Compute Returns and Advantages ==========
        with torch.no_grad():
            obs_tensor = obs_dict_to_tensor(obs, device)
            if is_single_stage:
                _, _, last_value, _ = policy(obs_tensor)
            else:
                _, last_value, _ = policy(obs_tensor)
            last_value = last_value.item()
        
        buffer.compute_returns_and_advantage(last_value, done)
        
        # ========== PPO Update ==========
        policy.train()
        
        # 记录指标
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        for epoch in range(args.update_epochs):
            for batch in buffer.get(minibatch_size):
                batch_obs = {
                    'uav_feats': batch.uav_feats,
                    'uav_mask': batch.uav_mask,
                    'sens_feats': batch.sens_feats,
                    'sens_mask': batch.sens_mask,
                    'global_feats': batch.global_feats,
                }

                if is_single_stage:
                    # ---------- single-stage PPO update ----------
                    from torch.distributions import Categorical
                    pair_logits, pair_mask, values, cache = policy(batch_obs)
                    values = values.squeeze(-1) if values.dim() > 1 else values

                    pair_probs, _ = masked_softmax(pair_logits, pair_mask, dim=-1)
                    pair_dist = Categorical(probs=pair_probs)

                    M1 = batch.sens_feats.size(1) + 1
                    pair_idx = SingleStagePairPolicy.encode_pair_action(
                        batch.uav_actions, batch.sens_actions, M1
                    )
                    new_log_prob = pair_dist.log_prob(pair_idx)
                    entropy_val = pair_dist.entropy()
                    entropy_loss = -entropy_val.mean()

                else:
                    # ---------- factorized PPO update ----------
                    from torch.distributions import Categorical
                    uav_logits, values, cache = policy(batch_obs)
                    values = values.squeeze(-1) if values.dim() > 1 else values

                    uav_probs, _ = masked_softmax(uav_logits, batch.uav_mask, dim=-1)
                    uav_dist = Categorical(probs=uav_probs)
                    new_uav_log_prob = uav_dist.log_prob(batch.uav_actions)
                    uav_entropy = uav_dist.entropy()

                    uav_e = cache['uav_e']
                    sens_e = cache['sens_e']
                    B = uav_e.size(0)
                    idx = batch.uav_actions.view(B, 1, 1).expand(B, 1, uav_e.size(-1))
                    chosen_uav_e = uav_e.gather(1, idx).squeeze(1)

                    smore_kw = _smore_stage2_kwargs(policy, batch, batch.uav_actions)
                    sens_logits, sens_mask_all = policy.conditional_sens_logits(
                        chosen_uav_e, sens_e, batch.sens_mask, batch.global_feats,
                        **smore_kw,
                    )
                    sens_probs, _ = masked_softmax(sens_logits, sens_mask_all, dim=-1)
                    sens_dist = Categorical(probs=sens_probs)
                    new_sens_log_prob = sens_dist.log_prob(batch.sens_actions)
                    sens_entropy = sens_dist.entropy()

                    new_log_prob = new_uav_log_prob + new_sens_log_prob
                    entropy_loss = -(uav_entropy.mean() + sens_entropy.mean())

                # ---- shared PPO loss computation ----
                old_log_prob = batch.old_uav_log_probs + batch.old_sens_log_probs

                advantages = batch.advantages
                if args.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                policy_loss, clip_fraction = compute_ppo_loss(
                    old_log_prob, new_log_prob, advantages, args.clip_coef
                )

                if args.clip_vloss:
                    value_loss = compute_value_loss(
                        values, batch.old_values, batch.returns, args.clip_coef
                    )
                else:
                    value_loss = compute_value_loss(values, batch.old_values, batch.returns, None)

                loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss

                # DyPS 可选 CVAE 辅助损失（ELBO 监督任务/组潜变量）
                if (
                    is_single_stage
                    and args.policy_variant == "dyps"
                    and args.dyps_cvae_coef > 0.0
                    and "cvae_mu" in cache
                ):
                    cvae_loss = DyPSHierarchicalPolicy.cvae_loss(
                        sens_feats=batch.sens_feats,
                        recon=cache["cvae_recon"],
                        mu=cache["cvae_mu"],
                        logvar=cache["cvae_logvar"],
                        sens_mask=batch.sens_mask,
                    )
                    loss = loss + args.dyps_cvae_coef * cvae_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction.item())
        
        # ========== Logging ==========
        if update % args.log_interval == 0:
            # 计算 explained variance
            y_pred = buffer.values[:buffer.pos]
            y_true = buffer.returns[:buffer.pos]
            explained_var = explained_variance(y_pred, y_true)
            
            # 计算 SPS
            sps = int(global_step / (time.time() - start_time))
            
            # print(f"Update {update}/{num_updates}: "
            #       f"global_step={global_step}, "
            #       f"pg_loss={np.mean(pg_losses):.4f}, "
            #       f"v_loss={np.mean(value_losses):.4f}, "
            #       f"entropy={-np.mean(entropy_losses):.4f}, "
            #       f"SPS={sps}")
            
            # TensorBoard
            writer.add_scalar("losses/policy_loss", np.mean(pg_losses), global_step)
            writer.add_scalar("losses/value_loss", np.mean(value_losses), global_step)
            writer.add_scalar("losses/entropy", -np.mean(entropy_losses), global_step)
            writer.add_scalar("losses/clip_fraction", np.mean(clip_fractions), global_step)
            writer.add_scalar("charts/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
        
        # ========== Save Model ==========
        if update % args.save_interval == 0:
            os.makedirs(args.model_save_dir, exist_ok=True)
            save_path = os.path.join(args.model_save_dir, f"pref_policy_{update}.pt")
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'global_step': global_step,
                'policy_variant': args.policy_variant,
            }, save_path)
            print(f"[Saved] {save_path}")
        
        # ========== Validation (test 见训练结束后的单次评估) ==========
        if args.eval_interval > 0 and update % args.eval_interval == 0:
            old_allowed = env.base_env.allowed_dates
            
            # --- Validation ---
            if val_dates:
                val_subset = val_dates[:args.max_eval_episodes] if args.max_eval_episodes > 0 else val_dates
                val_metrics = evaluate_on_dates(
                    policy=policy, macro_env=env, dates=val_subset,
                    region_bounds=region_bounds_for_eval, env_config=env_config,
                    args=args, device=device, seed=args.seed + 10000,
                )
                # print(f"[Val]  update={update}, "
                #       f"objective={val_metrics['avg_objective']:.2f}±{val_metrics['std_objective']:.2f}, "
                #       f"orders={val_metrics['order_completion_rate']:.1%}, "
                #       f"sensing={val_metrics['sensing_completion_rate']:.1%}, "
                #       f"delivery_profit={val_metrics['avg_delivery_profit']:.2f}, "
                #       f"sensing_entropy={val_metrics['avg_sensing_entropy']:.2f}, "
                #       f"energy_cost={val_metrics['avg_energy_cost']:.4f}")
                # ---- 业务 objective（论文指标）----
                writer.add_scalar("val/objective",               val_metrics['avg_objective'],           global_step)
                writer.add_scalar("val/sensing_entropy",         val_metrics['avg_sensing_entropy'],     global_step)
                writer.add_scalar("val/delivery_profit",         val_metrics['avg_delivery_profit'],     global_step)
                writer.add_scalar("val/energy_cost",             val_metrics['avg_energy_cost'],         global_step)
                writer.add_scalar("val/sensing_entropy_per_energy", val_metrics['sensing_entropy_per_energy'], global_step)
                writer.add_scalar("val/delivery_profit_per_energy", val_metrics['delivery_profit_per_energy'], global_step)
                # ---- 完成率 & 训练监控 ----
                writer.add_scalar("val/order_completion_rate",   val_metrics['order_completion_rate'],  global_step)
                writer.add_scalar("val/sensing_completion_rate", val_metrics['sensing_completion_rate'], global_step)
                writer.add_scalar("val/train_reward",            val_metrics['avg_train_reward'],        global_step)

                if val_metrics['avg_objective'] > best_val_reward:
                    best_val_reward = val_metrics['avg_objective']
                    best_path = os.path.join(args.model_save_dir, "best_val_model.pt")
                    os.makedirs(args.model_save_dir, exist_ok=True)
                    torch.save({
                        'policy_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'update': update, 'global_step': global_step,
                        'val_metrics': val_metrics,
                        'policy_variant': args.policy_variant,
                    }, best_path)
                    print(f"[Best Val Model Saved] reward={best_val_reward:.2f} -> {best_path}")
            
            env.base_env.allowed_dates = old_allowed
            obs = env.reset()
            # 评估打断了正在进行的训练 episode，环境已重置为新 episode，
            # 必须同步清零训练的 episode 级累加器，否则旧残留值会污染下一个 episode 的统计。
            episode_return          = 0.0
            episode_sensing_entropy = 0.0
            episode_delivery_profit = 0.0
            episode_energy_cost     = 0.0
            episode_sensing_timeout = 0.0
            episode_order_timeout   = 0.0
            episode_shaping         = 0.0
    
    # 保存最终模型
    os.makedirs(args.model_save_dir, exist_ok=True)
    final_path = os.path.join(args.model_save_dir, f"{run_name}_final.pt")
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update': num_updates,
        'global_step': global_step,
        'policy_variant': args.policy_variant,
    }, final_path)
    print(f"[Final Model Saved] {final_path}")
    
    # ========== Test（训练全部结束后单次评估，避免与 val 同期重复跑）==========
    if test_dates:
        old_allowed = env.base_env.allowed_dates
        test_subset = test_dates[:args.max_eval_episodes] if args.max_eval_episodes > 0 else test_dates
        test_metrics = evaluate_on_dates(
            policy=policy, macro_env=env, dates=test_subset,
            region_bounds=region_bounds_for_eval, env_config=env_config,
            args=args, device=device, seed=args.seed + 20000,
        )
        writer.add_scalar("test/objective",               test_metrics['avg_objective'],           global_step)
        writer.add_scalar("test/sensing_entropy",         test_metrics['avg_sensing_entropy'],     global_step)
        writer.add_scalar("test/delivery_profit",         test_metrics['avg_delivery_profit'],     global_step)
        writer.add_scalar("test/energy_cost",             test_metrics['avg_energy_cost'],         global_step)
        writer.add_scalar("test/sensing_entropy_per_energy", test_metrics['sensing_entropy_per_energy'], global_step)
        writer.add_scalar("test/delivery_profit_per_energy", test_metrics['delivery_profit_per_energy'], global_step)
        writer.add_scalar("test/order_completion_rate",   test_metrics['order_completion_rate'],  global_step)
        writer.add_scalar("test/sensing_completion_rate", test_metrics['sensing_completion_rate'], global_step)
        writer.add_scalar("test/train_reward",            test_metrics['avg_train_reward'],        global_step)
        env.base_env.allowed_dates = old_allowed
        print(f"[Test @ end] objective={test_metrics['avg_objective']:.2f}, "
              f"orders={test_metrics['order_completion_rate']:.1%}, "
              f"sensing={test_metrics['sensing_completion_rate']:.1%}")
    
    # 清理
    env.close()
    writer.close()
    
    if args.track:
        import wandb
        wandb.finish()
    
    print("\n[Training Complete]")


def evaluate_only(args: Args):
    """加载已训练的 Preference PPO 模型，在指定数据子集上评估"""
    env_config = get_config_template(args.config_template)

    # CLI 优先：仅当 CLI 未显式设置时才从 JSON 覆盖
    _defaults = Args()
    for _k in ['num_uavs', 'K', 'sensing_weight',
               'd_emb', 'd_hidden', 'num_sensing_points', 'seed']:
        if _k in env_config and getattr(args, _k) == getattr(_defaults, _k):
            setattr(args, _k, env_config[_k])

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = env_config.get('dataset_path', 'datasets')
    orders_filename = env_config.get('orders_filename', 'hangzhou_region0_101_MayToJuly.csv')
    csv_path = os.path.join(dataset_path, orders_filename)
    dataset_manager = DatasetManager(csv_path, config_template=args.config_template)

    region_id = env_config.get('region_id', 0)
    split_path = env_config.get('split_path', args.split_path)
    train_dates, val_dates, test_dates = DatasetManager.get_split_dates(
        dataset_manager=dataset_manager,
        region_id=str(region_id),
        split_path=split_path,
    )
    split_map = {'train': train_dates, 'val': val_dates, 'test': test_dates}
    eval_dates = split_map[args.eval_split]
    if args.max_eval_episodes > 0 and args.max_eval_episodes < len(eval_dates):
        eval_dates = eval_dates[:args.max_eval_episodes]
    print(f"[Eval] split={args.eval_split}, dates={len(eval_dates)}")

    env = make_env(args, env_config, dataset_manager)

    d_uav, d_sens, d_global = 9, 8, 4
    if args.policy_variant == "d2sn":
        policy = D2SNSingleStepPairPolicy(
            duav=d_uav, dsens=d_sens, dglobal=d_global,
            demb=args.d_emb, hid=args.d_hidden,
            max_uavs=env.num_uavs,
            max_sens=env.max_sensing_points,
        ).to(device)
    elif args.policy_variant == "dyps":
        policy = DyPSHierarchicalPolicy(
            duav=d_uav, dsens=d_sens, dglobal=d_global,
            dmodel=args.d_emb, hid=args.d_hidden,
            max_uavs=env.num_uavs,
            max_sens=env.max_sensing_points,
        ).to(device)
    elif args.policy_variant == "single_stage_pair":
        policy = SingleStagePairPolicy(
            duav=d_uav, dsens=d_sens, dglobal=d_global,
            demb=args.d_emb, hid=args.d_hidden,
            max_uavs=env.num_uavs,
            max_sens=env.max_sensing_points,
        ).to(device)
    elif args.policy_variant == "smore":
        policy = SMOREBaseline(
            duav=d_uav, dsens=d_sens, dglobal=d_global,
            demb=args.d_emb, hid=args.d_hidden,
            max_uavs=env.num_uavs,
            max_sens=env.max_sensing_points,
        ).to(device)
    elif args.policy_variant == "deco":
        policy = DECOActorCritic(
            duav=d_uav, dsens=d_sens, dglobal=d_global,
            demb=args.d_emb, hid=args.d_hidden,
            max_sens=env.max_sensing_points,
        ).to(device)
    else:
        policy = PreferenceActorCritic(
            duav=d_uav, dsens=d_sens, dglobal=d_global,
            demb=args.d_emb, hid=args.d_hidden,
            max_sens=env.max_sensing_points,
        ).to(device)

    model_path = args.eval_model_path
    if model_path is None:
        model_path = os.path.join(args.model_save_dir, "best_val_model.pt")
    print(f"[Loading Model] {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    policy.load_state_dict(ckpt['policy_state_dict'])
    if 'val_metrics' in ckpt:
        print(f"  (saved val reward: {ckpt['val_metrics'].get('avg_reward', '?')})")

    region_bounds = env.base_env.region_bounds

    metrics = evaluate_on_dates(
        policy=policy,
        macro_env=env,
        dates=eval_dates,
        region_bounds=region_bounds,
        env_config=env_config,
        args=args,
        device=device,
        seed=args.seed,
    )


    env.close()
    return metrics


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.eval_only:
        evaluate_only(args)
    else:
        train(args)

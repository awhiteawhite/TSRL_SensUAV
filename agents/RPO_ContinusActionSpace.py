# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from environments.uav_environment import UAVEnvironment
from config import get_config_template
from datasets.dataset_manager import DatasetManager


@dataclass
class Args:
    # 配置模板参数
    config_template: str = "default"
    """使用的配置模板名称 ("default", "debug", "fast_training", "production")"""

    # 基础参数
    exp_name: str = "RPO_ContinusActionSpace"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # 环境参数
    env_id: str = "UAV-v0"
    """the id of the environment"""

    # 算法参数
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    max_episodes: int = 10000
    """maximum number of episodes (daily episodes) to run before stopping"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""

    # 多UAV参数
    num_uavs: int = 1
    """the number of UAVs in the environment"""

    # 运行时计算的参数
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma, config=None, num_uavs=1,
             dataset_manager: DatasetManager = None, region_id: int = 0, allowed_dates=None):
    def thunk():
        if env_id == "UAV-v0":
            # Create UAV environment with config
            # 使用real_data.json中的配置启用daily episode模式（如果配置中有该字段）
            episode_sampling_mode = config.get("episode_sampling_mode", "random") if config is not None else "random"
            # 合并allowed_dates到config中
            env_config = config.copy() if config else {}
            if allowed_dates is not None:
                env_config["allowed_dates"] = allowed_dates
            env = UAVEnvironment(
                config=env_config,
                num_uavs=num_uavs,
                dataset_manager=dataset_manager,
                episode_sampling_mode=episode_sampling_mode,
                region_id=region_id,
            )
        else:
            # Original gym environments
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)

        # Apply common wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Only apply FlattenObservation if it's a Dict observation space
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = gym.wrappers.FlattenObservation(env)

        # For continuous action spaces, apply ClipAction
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)

        # Normalize observation if it's continuous
        if isinstance(env.observation_space, gym.spaces.Box):
            env = gym.wrappers.NormalizeObservation(env)
            # Note: Skip TransformObservation for now to avoid version compatibility issues

        # # Normalize rewards (skip TransformReward to avoid compatibility issues)
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class UAVAgent(nn.Module):
    """
    Agent class for UAV environment with speed control action space.
    
    Action space: speed ratio [0, 1] for each UAV
    - agent输出速度比例，乘以max_speed得到实际速度（m/min）
    - 飞行方向由环境根据任务状态自动决定
    """
    def __init__(self, envs, rpo_alpha):
        super().__init__()
        self.rpo_alpha = rpo_alpha

        # Get action dimension from environment (supports multi-UAV: num_uavs)
        # 每个UAV只有1个动作维度（速度比例）
        action_dim = np.array(envs.single_action_space.shape).prod()

        # Critic network (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor network for continuous action space
        obs_dim = np.array(envs.single_observation_space.shape).prod()

        # Speed control: speed ratio for each UAV
        # action_dim = num_uavs (each UAV has 1 action dimension: speed ratio)
        # 输出通过 Sigmoid 映射到 [0, 1]，代表速度比例
        self.actor_speed_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_speed_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        处理连续动作的策略网络（支持多UAV）

        Args:
            x: 观测，形状 [batch, obs_dim]
            action: 可选的旧动作张量 [batch, action_dim]，用于RPO/PPO更新时重新计算log prob
                   action_dim = num_uavs (每个UAV有1个动作维度: speed_ratio)
        
        Returns:
            speed_action: 速度比例动作 [batch, num_uavs]，范围约在[0, 1]
            total_logprob: 动作的log概率
            total_entropy: 策略熵
            value: 状态价值
        """
        # Get speed action (continuous)
        speed_mean = self.actor_speed_mean(x)
        speed_logstd = self.actor_speed_logstd.expand_as(speed_mean)
        speed_std = torch.exp(speed_logstd)
        speed_probs = Normal(speed_mean, speed_std)

        if action is None:
            # Sample actions during data collection
            # 采样后通过sigmoid映射到[0, 1]范围
            raw_action = speed_probs.sample()
            speed_action = torch.sigmoid(raw_action)
        else:
            # Use provided action (for PPO update)
            # 这里 action 已经是映射后的速度比例 [batch, num_uavs]
            speed_action = action
            # 反映射回原始空间以计算log prob
            # 注意：这里使用的是原始采样空间，不是sigmoid后的空间
            # 为了简化，我们直接在原始空间计算log prob
            raw_action = torch.logit(torch.clamp(speed_action, 1e-6, 1 - 1e-6))

            # 如果提供的动作 batch 大小比当前 x 小，子集对齐（用于minibatch）
            if raw_action.shape[0] != speed_mean.shape[0]:
                batch_size = raw_action.shape[0]
                speed_mean = speed_mean[:batch_size]
                speed_std = speed_std[:batch_size]
                speed_probs = Normal(speed_mean, speed_std)

        # Calculate log probabilities (在原始空间计算)
        speed_logprob = speed_probs.log_prob(raw_action).sum(dim=1)
        total_logprob = speed_logprob

        # Calculate entropy
        speed_entropy = speed_probs.entropy().sum(dim=1)
        total_entropy = speed_entropy

        return speed_action, total_logprob, total_entropy, self.critic(x)


def evaluate_agent(agent, test_dates, dataset_manager, env_config, 
                   num_uavs, device, num_episodes=10, deterministic=True,
                   writer=None, wandb_run=None, global_step=0):
    """
    在测试集上评估agent性能，并记录到wandb和tensorboard
    
    Args:
        agent: 训练好的agent
        test_dates: 测试集日期列表
        dataset_manager: 数据集管理器
        env_config: 环境配置
        num_uavs: UAV数量
        device: 设备
        num_episodes: 评估的episode数量
        deterministic: 是否使用确定性策略（取mean而不是sample）
        writer: tensorboard writer（可选）
        wandb_run: wandb run对象（可选）
        global_step: 当前训练步数
    
    Returns:
        eval_results: 评估结果字典
    """
    agent.eval()
    
    # 创建评估环境（使用测试集日期）
    eval_envs = gym.vector.SyncVectorEnv([
        make_env(
            env_id="UAV-v0",
            idx=0,
            capture_video=False,
            run_name="eval",
            gamma=env_config.get("gamma", 0.99),
            config=env_config,
            num_uavs=num_uavs,
            dataset_manager=dataset_manager,
            region_id=env_config.get("region_id", 0),
            allowed_dates=test_dates,  # 限制为测试集日期
        )
    ])
    
    episode_returns = []
    episode_lengths = []
    completed_orders = []
    timed_out_orders = []
    total_orders_in_episodes = []
    
    for episode_idx in range(num_episodes):
        obs, _ = eval_envs.reset()
        obs = torch.Tensor(obs).to(device)
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_completed = 0
        episode_timed_out = 0
        episode_total_orders = 0
        
        while not done:
            with torch.no_grad():
                if deterministic:
                    # 使用确定性策略（取mean并通过sigmoid映射到[0,1]）
                    speed_mean = agent.actor_speed_mean(obs)
                    action = torch.sigmoid(speed_mean)  # 映射到[0, 1]作为速度比例
                else:
                    # 使用随机策略
                    action, _, _, _ = agent.get_action_and_value(obs)
            
            obs, reward, terminations, truncations, infos = eval_envs.step(
                action.cpu().numpy()
            )
            obs = torch.Tensor(obs).to(device)
            done = terminations[0] or truncations[0]
            episode_return += reward[0]
            episode_length += 1
        
        # 提取订单完成信息
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info:
                    episode_completed = info.get("total_orders_completed", 0)
                    episode_timed_out = info.get("total_orders_timed_out", 0)
                    episode_total_orders = info.get("total_orders_in_episode", 0)
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        completed_orders.append(episode_completed)
        timed_out_orders.append(episode_timed_out)
        total_orders_in_episodes.append(episode_total_orders)
        
        # 打印每个episode的订单统计
        if episode_total_orders > 0:
            completion_rate = episode_completed / episode_total_orders * 100
            print(f"  Episode {episode_idx + 1}/{num_episodes}: 总订单数={episode_total_orders}, "
                  f"已完成={episode_completed}, 超时={episode_timed_out}, 完成率={completion_rate:.2f}%")
    
    eval_envs.close()
    agent.train()
    
    # 计算统计信息
    mean_completed = np.mean(completed_orders) if completed_orders else 0
    mean_timed_out = np.mean(timed_out_orders) if timed_out_orders else 0
    mean_total_orders = np.mean(total_orders_in_episodes) if total_orders_in_episodes else 0
    completion_rate = mean_completed / (mean_completed + mean_timed_out + 1e-8)
    # 基于总订单数的完成率
    overall_completion_rate = (mean_completed / mean_total_orders * 100) if mean_total_orders > 0 else 0.0
    
    eval_results = {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "min_return": np.min(episode_returns),
        "max_return": np.max(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "mean_completed_orders": mean_completed,
        "mean_timed_out_orders": mean_timed_out,
        "mean_total_orders": mean_total_orders,
        "completion_rate": completion_rate,
        "overall_completion_rate": overall_completion_rate,
        "episode_returns": episode_returns,
    }
    
    # TensorBoard记录
    if writer:
        writer.add_scalar("eval/episodic_return", eval_results["mean_return"], global_step)
        writer.add_scalar("eval/episodic_return_std", eval_results["std_return"], global_step)
        writer.add_scalar("eval/episodic_length", eval_results["mean_length"], global_step)
        writer.add_scalar("eval/mean_total_orders", eval_results["mean_total_orders"], global_step)
        writer.add_scalar("eval/mean_completed_orders", eval_results["mean_completed_orders"], global_step)
        writer.add_scalar("eval/mean_timed_out_orders", eval_results["mean_timed_out_orders"], global_step)
        writer.add_scalar("eval/completion_rate", eval_results["completion_rate"], global_step)
        writer.add_scalar("eval/overall_completion_rate", eval_results["overall_completion_rate"], global_step)
    
    # WandB直接记录
    if wandb_run:
        wandb.log({
            "eval/mean_return": eval_results["mean_return"],
            "eval/std_return": eval_results["std_return"],
            "eval/min_return": eval_results["min_return"],
            "eval/max_return": eval_results["max_return"],
            "eval/mean_length": eval_results["mean_length"],
            "eval/mean_total_orders": eval_results["mean_total_orders"],
            "eval/mean_completed_orders": eval_results["mean_completed_orders"],
            "eval/mean_timed_out_orders": eval_results["mean_timed_out_orders"],
            "eval/completion_rate": eval_results["completion_rate"],
            "eval/overall_completion_rate": eval_results["overall_completion_rate"],
        }, step=global_step)
        
        # 记录分布
        wandb.log({
            "eval/return_distribution": wandb.Histogram(episode_returns),
            "eval/length_distribution": wandb.Histogram(episode_lengths),
        }, step=global_step)
        
        # 创建表格
        eval_table = wandb.Table(columns=[
            "episode", "return", "length", "total_orders", "completed_orders", "timed_out_orders"
        ], data=[
            [i, ret, length, total, comp, timeout] 
            for i, (ret, length, total, comp, timeout) in enumerate(zip(
                episode_returns, episode_lengths, total_orders_in_episodes, completed_orders, timed_out_orders
            ))
        ])
        wandb.log({"eval/detailed_results": eval_table}, step=global_step)
    
    return eval_results


if __name__ == "__main__":
    args = tyro.cli(Args)

    # 如果指定了配置模板，使用该模板更新参数
    if args.config_template != "default":
        from config import get_config_template
        template_config = get_config_template(args.config_template)

        # 更新 args 的相关字段
        for key, value in template_config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # 从配置获取环境配置
    from config import get_config_template
    env_config = get_config_template(args.config_template)

    # 基于配置创建DatasetManager（用于daily episode采样）
    dataset_path = env_config.get("dataset_path", "datasets")
    orders_filename = env_config.get("orders_filename", "hangzhou_region0_101_MayToJuly.csv")
    csv_path = os.path.join(dataset_path, orders_filename)
    region_id = env_config.get("region_id", 0)

    dataset_manager = DatasetManager(csv_path, config_template=args.config_template)
    
    # 数据集分割：训练集和测试集
    train_dates = None
    test_dates = None
    # 注意：配置已被扁平化，直接从 env_config 读取
    enable_test_set = env_config.get("enable_test_set", False)
    
    if enable_test_set:
        train_ratio = env_config.get("train_test_split_ratio", 0.8)
        split_date = env_config.get("split_date", None)
        train_dates, test_dates = dataset_manager.split_train_test_dates(
            region_id=str(region_id),
            train_ratio=train_ratio,
            split_date=split_date
        )
        print(f"[Dataset Split] Train dates: {len(train_dates)} ({train_dates[0] if train_dates else 'N/A'} to {train_dates[-1] if train_dates else 'N/A'})")
        print(f"[Dataset Split] Test dates: {len(test_dates)} ({test_dates[0] if test_dates else 'N/A'} to {test_dates[-1] if test_dates else 'N/A'})")
    else:
        print("[Dataset Split] Test set evaluation is disabled")
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        
        # 记录数据集分割信息到wandb
        if enable_test_set and train_dates and test_dates:
            wandb.config.update({
                "dataset/train_date_count": len(train_dates),
                "dataset/test_date_count": len(test_dates),
                "dataset/train_date_range": f"{train_dates[0]} to {train_dates[-1]}",
                "dataset/test_date_range": f"{test_dates[0]} to {test_dates[-1]}",
                "dataset/train_test_split": f"{len(train_dates)}/{len(test_dates)}",
            })
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # 训练环境使用训练集日期（如果启用了测试集）
    train_allowed_dates = train_dates if enable_test_set else None
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                config=env_config,
                num_uavs=args.num_uavs,
                dataset_manager=dataset_manager,
                region_id=region_id,
                allowed_dates=train_allowed_dates,
            )
            for i in range(args.num_envs)
        ]
    )
    # Check action space compatibility: 现在统一要求连续动作空间
    assert isinstance(envs.single_action_space, gym.spaces.Box), "environment must have continuous (Box) action space"

    agent = UAVAgent(envs, args.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    # 连续动作空间的统一存储
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    episode_count = 0
    reached_max_episodes = False
    last_train_return = 0.0  # 用于记录最后的训练集return，用于评估对比
    
    # 评估相关配置
    eval_freq = env_config.get("eval_freq", 100000)
    num_eval_episodes = env_config.get("num_eval_episodes", 10)
    best_eval_return = float('-inf')
    # 计算评估间隔（基于 update 次数），避免 global_step % eval_freq 永远不为0的问题
    eval_interval_updates = max(1, eval_freq // args.batch_size)
    print(f"[Eval Config] Will evaluate every {eval_interval_updates} updates (eval_freq={eval_freq}, batch_size={args.batch_size})")
    wandb_run = None
    if args.track:
        import wandb
        wandb_run = wandb.run

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            # 连续动作环境：直接存储并发送到环境
            actions[step] = action
            env_action = action.cpu().numpy()

            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(env_action)
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_count += 1
                        ep_ret = info["episode"]["r"]
                        ep_len = info["episode"]["l"]
                        date_str = info.get("date_str", "unknown")
                        print(f"episode={episode_count}, date={date_str}, global_step={global_step}, episodic_return={ep_ret}")
                        
                        # 打印订单统计信息
                        if "total_orders_in_episode" in info:
                            total_orders = info["total_orders_in_episode"]
                            completed_orders = info.get("total_orders_completed", 0)
                            timed_out_orders = info.get("total_orders_timed_out", 0)
                            completion_rate = (completed_orders / total_orders * 100) if total_orders > 0 else 0.0
                            print(f"  -> 订单统计: 总订单数={total_orders}, 已完成={completed_orders}, 超时={timed_out_orders}, 完成率={completion_rate:.2f}%")
                        
                        writer.add_scalar("train/episodic_return", ep_ret, global_step)
                        writer.add_scalar("train/episodic_length", ep_len, global_step)
                        writer.add_scalar("charts/episode_index", episode_count, global_step)
                        
                        # WandB直接记录训练指标（使用episode_count作为横轴）
                        if wandb_run:
                            log_dict = {
                                "train/episodic_return": ep_ret,
                                "train/episodic_length": ep_len,
                            }
                            # 添加订单统计到WandB
                            if "total_orders_in_episode" in info:
                                log_dict["train/total_orders_in_episode"] = info["total_orders_in_episode"]
                                log_dict["train/total_orders_completed"] = info.get("total_orders_completed", 0)
                                log_dict["train/total_orders_timed_out"] = info.get("total_orders_timed_out", 0)
                                log_dict["train/completion_rate"] = completion_rate
                            wandb.log(log_dict, step=episode_count)

                # 检查是否达到最大episode数量
                if episode_count >= args.max_episodes:
                    reached_max_episodes = True
                    break

        if reached_max_episodes:
            print(f"Reached max_episodes={args.max_episodes}, stopping training loop.")
            break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # 连续动作格式：直接取子批次动作
                mb_actions = b_actions[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], mb_actions)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # 评估（如果启用了测试集）
        # 使用基于 update 次数的条件，避免 global_step % eval_freq 永远不为0的问题
        if enable_test_set and test_dates and update % eval_interval_updates == 0:
            print(f"\n[Evaluation] Evaluating at step {global_step}...")
            eval_results = evaluate_agent(
                agent=agent,
                test_dates=test_dates,
                dataset_manager=dataset_manager,
                env_config=env_config,
                num_uavs=args.num_uavs,
                device=device,
                num_episodes=num_eval_episodes,
                deterministic=True,
                writer=writer,
                wandb_run=wandb_run,
                global_step=global_step
            )
            
            print(f"[Evaluation] Mean Return: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")
            print(f"[Evaluation] Mean Total Orders in Episode: {eval_results['mean_total_orders']:.2f}")
            print(f"[Evaluation] Mean Completed Orders: {eval_results['mean_completed_orders']:.2f}")
            print(f"[Evaluation] Mean Timed Out Orders: {eval_results['mean_timed_out_orders']:.2f}")
            print(f"[Evaluation] Overall Completion Rate (completed/total): {eval_results['overall_completion_rate']:.2f}%")
            print(f"[Evaluation] Completion Rate (completed/(completed+timed_out)): {eval_results['completion_rate']:.2%}")
            
            # 获取最近训练集指标进行对比
            if wandb_run:
                # 计算过拟合指标
                # 使用最后记录的训练集return（如果存在）
                train_return = last_train_return if 'last_train_return' in locals() else 0
                if train_return > 0:
                    wandb.log({
                        "comparison/train_vs_eval_return": eval_results["mean_return"] - train_return,
                        "comparison/overfitting_score": train_return - eval_results["mean_return"],
                    }, step=global_step)
            
            # 保存最佳模型
            save_best_model = env_config.get("save_best_model", False)
            if save_best_model and eval_results["mean_return"] > best_eval_return:
                best_eval_return = eval_results["mean_return"]
                model_save_dir = env_config.get("model_save_dir", "models")
                os.makedirs(model_save_dir, exist_ok=True)
                model_path = os.path.join(model_save_dir, f"best_model_step_{global_step}.pt")
                torch.save({
                    'agent_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'eval_return': eval_results["mean_return"],
                }, model_path)
                
                print(f"[Evaluation] Saved best model with return {best_eval_return:.2f} to {model_path}")
                
                if wandb_run:
                    wandb.log({
                        "best_eval/return": best_eval_return,
                        "best_eval/step": global_step,
                    }, step=global_step)

    # 训练结束后保存最终模型（不依赖 eval；用于单日数据等无法做 train/test 切分的场景）
    save_best_model = env_config.get("save_best_model", False)
    if save_best_model:
        model_save_dir = env_config.get("model_save_dir", "models")
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, f"final_model_step_{global_step}.pt")
        torch.save({
            'agent_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_path)
        print(f"[Training] Saved final model at step {global_step} to {model_path}")

    envs.close()
    writer.close()
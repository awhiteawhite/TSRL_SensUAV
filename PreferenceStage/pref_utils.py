"""
Preference Stage 工具函数

包含：
- 采样函数
- 观察转换函数
- 感知点生成函数
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    带 mask 的 softmax
    
    Args:
        logits: 原始 logits
        mask: 有效性 mask (1=valid, 0=invalid)
        dim: softmax 维度
    
    Returns:
        probs: softmax 概率
        masked_logits: mask 后的 logits
    """
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(mask == 0, neg_inf)
    probs = F.softmax(masked_logits, dim=dim)
    return probs, masked_logits


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    带 mask 的均值
    
    Args:
        x: 输入 tensor (..., L, D)
        mask: mask (..., L) with 1/0
        dim: 求和维度
    
    Returns:
        mean: 均值
    """
    mask = mask.unsqueeze(-1)  # (..., L, 1)
    x = x * mask
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return x.sum(dim=dim) / denom


def sample_uav_action(
    uav_logits: torch.Tensor,
    uav_mask: torch.Tensor,
    deterministic: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    采样 UAV 选择动作
    
    Args:
        uav_logits: (B, N) UAV 选择的 logits
        uav_mask: (B, N) UAV 有效性 mask
        deterministic: 是否确定性选择（选择概率最高的）
    
    Returns:
        action: (B,) 选择的 UAV ID
        log_prob: (B,) log 概率
        entropy: (B,) 熵
        probs: (B, N) 概率分布
    """
    probs, masked_logits = masked_softmax(uav_logits, uav_mask, dim=-1)
    dist = Categorical(probs=probs)
    
    if deterministic:
        action = probs.argmax(dim=-1)
    else:
        action = dist.sample()
    
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    
    return action, log_prob, entropy, probs


def sample_sens_action(
    sens_logits: torch.Tensor,
    sens_mask: torch.Tensor,
    deterministic: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    采样 sensing point 选择动作（包含 SKIP 选项）
    
    Args:
        sens_logits: (B, M+1) sensing 选择的 logits，索引 0 是 SKIP
        sens_mask: (B, M+1) 有效性 mask（SKIP 总是有效）
        deterministic: 是否确定性选择
    
    Returns:
        action: (B,) 选择的 sensing ID (0 = SKIP, 1..M = 实际感知点)
        log_prob: (B,) log 概率
        entropy: (B,) 熵
        probs: (B, M+1) 概率分布
    """
    probs, masked_logits = masked_softmax(sens_logits, sens_mask, dim=-1)
    dist = Categorical(probs=probs)
    
    if deterministic:
        action = probs.argmax(dim=-1)
    else:
        action = dist.sample()
    
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    
    return action, log_prob, entropy, probs


def sample_pair_action(
    pair_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    deterministic: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample from a flat (uav, sensing) pair distribution.

    Args:
        pair_logits: (B, N*(M+1))
        pair_mask:   (B, N*(M+1))   1=valid, 0=invalid
        deterministic: argmax if True

    Returns:
        action:   (B,) pair index
        log_prob: (B,)
        entropy:  (B,)
        probs:    (B, N*(M+1))
    """
    probs, _ = masked_softmax(pair_logits, pair_mask, dim=-1)
    dist = Categorical(probs=probs)

    if deterministic:
        action = probs.argmax(dim=-1)
    else:
        action = dist.sample()

    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return action, log_prob, entropy, probs


def obs_dict_to_tensor(
    obs: Dict[str, np.ndarray],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    将 numpy 格式的观察字典转换为 PyTorch tensor
    
    Args:
        obs: numpy 格式的观察字典
        device: 目标设备
    
    Returns:
        tensor 格式的观察字典（添加 batch 维度）
    """
    return {
        'uav_feats': torch.FloatTensor(obs['uav_feats']).unsqueeze(0).to(device),
        'uav_mask': torch.FloatTensor(obs['uav_mask']).unsqueeze(0).to(device),
        'sens_feats': torch.FloatTensor(obs['sens_feats']).unsqueeze(0).to(device),
        'sens_mask': torch.FloatTensor(obs['sens_mask']).unsqueeze(0).to(device),
        'global_feats': torch.FloatTensor(obs['global_feats']).unsqueeze(0).to(device),
    }


def batch_obs_dict_to_tensor(
    obs_list: List[Dict[str, np.ndarray]],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    将多个观察字典批量转换为 PyTorch tensor
    
    Args:
        obs_list: 观察字典列表
        device: 目标设备
    
    Returns:
        batch 格式的 tensor 字典
    """
    return {
        'uav_feats': torch.FloatTensor(np.stack([o['uav_feats'] for o in obs_list])).to(device),
        'uav_mask': torch.FloatTensor(np.stack([o['uav_mask'] for o in obs_list])).to(device),
        'sens_feats': torch.FloatTensor(np.stack([o['sens_feats'] for o in obs_list])).to(device),
        'sens_mask': torch.FloatTensor(np.stack([o['sens_mask'] for o in obs_list])).to(device),
        'global_feats': torch.FloatTensor(np.stack([o['global_feats'] for o in obs_list])).to(device),
    }


def generate_sensing_points(
    region_bounds: Tuple[float, float, float, float],
    num_points: int,
    time_window_minutes: Tuple[int, int] = (30, 120),
    priority_range: Tuple[float, float] = (0.5, 1.0),
    seed: Optional[int] = None
) -> List[Dict]:
    """
    生成随机感知点配置
    
    Args:
        region_bounds: (min_lat, max_lat, min_lng, max_lng)
        num_points: 感知点数量
        time_window_minutes: (min, max) 时间窗长度范围
        priority_range: (min, max) 优先级范围
        seed: 随机种子
    
    Returns:
        感知点配置列表
    """
    if seed is not None:
        np.random.seed(seed)
    
    min_lat, max_lat, min_lng, max_lng = region_bounds
    
    sensing_points = []
    for i in range(num_points):
        lat = np.random.uniform(min_lat, max_lat)
        lng = np.random.uniform(min_lng, max_lng)
        priority = np.random.uniform(*priority_range)
        window = np.random.randint(time_window_minutes[0], time_window_minutes[1] + 1)
        
        sensing_points.append({
            'point_id': i,
            'location': (lat, lng),
            'priority': priority,
            'time_window_minutes': window,
            'data_value': priority,  # 数据价值与优先级挂钩
        })
    
    return sensing_points


def load_sensing_points_from_file(filepath: str) -> List[Dict]:
    """
    从文件加载感知点配置
    
    Args:
        filepath: JSON 文件路径
    
    Returns:
        感知点配置列表
    """
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 确保格式正确
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'sensing_points' in data:
        return data['sensing_points']
    else:
        raise ValueError(f"Invalid sensing points file format: {filepath}")


def compute_ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 PPO 策略损失
    
    Args:
        old_log_probs: 旧策略的 log 概率
        new_log_probs: 新策略的 log 概率
        advantages: 优势函数
        clip_range: PPO clip 范围
    
    Returns:
        loss: PPO 损失
        clip_fraction: clip 比例（用于监控）
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Clipped surrogate loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
    # Clip fraction for logging
    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float())
    
    return policy_loss, clip_fraction


def compute_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range_vf: Optional[float] = None
) -> torch.Tensor:
    """
    计算价值函数损失
    
    Args:
        values: 当前价值估计
        old_values: 旧的价值估计
        returns: 目标 returns
        clip_range_vf: 价值函数 clip 范围（可选）
    
    Returns:
        value_loss: 价值损失
    """
    if clip_range_vf is not None:
        # Clipped value loss
        values_clipped = old_values + torch.clamp(
            values - old_values, -clip_range_vf, clip_range_vf
        )
        value_loss_1 = (values - returns) ** 2
        value_loss_2 = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
    else:
        value_loss = 0.5 * ((values - returns) ** 2).mean()
    
    return value_loss


class RunningMeanStd:
    """
    Running mean and standard deviation
    
    用于奖励归一化等
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算 explained variance
    
    Args:
        y_pred: 预测值
        y_true: 真实值
    
    Returns:
        explained variance
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return np.nan
    return 1 - np.var(y_true - y_pred) / var_y

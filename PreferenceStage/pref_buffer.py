"""
Preference Rollout Buffer

Preference Stage 专用的 Rollout Buffer。
存储 macro step 的数据，用于 PPO 更新。

观察是 Dict 格式：
- uav_feats: (N, D_uav)
- uav_mask: (N,)
- sens_feats: (M, D_sens)
- sens_mask: (M,)
- global_feats: (D_global,)

动作是 (uav_id, sens_id) 元组。
"""

import numpy as np
import torch
from typing import Dict, Generator, Optional, NamedTuple


class PreferenceRolloutBufferSamples(NamedTuple):
    """Rollout buffer 采样数据"""
    uav_feats: torch.Tensor       # (batch, N, D_uav)
    uav_mask: torch.Tensor        # (batch, N)
    sens_feats: torch.Tensor      # (batch, M, D_sens)
    sens_mask: torch.Tensor       # (batch, M)
    global_feats: torch.Tensor    # (batch, D_global)
    uav_actions: torch.Tensor     # (batch,) int64
    sens_actions: torch.Tensor    # (batch,) int64
    old_uav_log_probs: torch.Tensor    # (batch,)
    old_sens_log_probs: torch.Tensor   # (batch,)
    advantages: torch.Tensor      # (batch,)
    returns: torch.Tensor         # (batch,)
    old_values: torch.Tensor      # (batch,)


class PreferenceRolloutBuffer:
    """
    Preference Stage 的 Rollout Buffer
    
    存储 macro step 的数据：
    - 观察（Dict 格式）
    - 动作（uav_id, sens_id）
    - 奖励、价值、log prob
    """
    
    def __init__(
        self,
        buffer_size: int,
        max_uavs: int,
        max_sensing_points: int,
        d_uav: int = 9,
        d_sens: int = 8,
        d_global: int = 4,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Args:
            buffer_size: buffer 大小（macro step 数量）
            max_uavs: 最大 UAV 数量
            max_sensing_points: 最大 sensing point 数量
            d_uav: UAV 特征维度
            d_sens: Sensing point 特征维度
            d_global: 全局特征维度
            device: 设备
            gamma: 折扣因子
            gae_lambda: GAE lambda
        """
        self.buffer_size = buffer_size
        self.max_uavs = max_uavs
        self.max_sensing_points = max_sensing_points
        self.d_uav = d_uav
        self.d_sens = d_sens
        self.d_global = d_global
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.pos = 0
        self.full = False
        
        # 初始化 buffer
        self.reset()
    
    def reset(self):
        """重置 buffer"""
        self.pos = 0
        self.full = False
        
        # 观察 buffer
        self.uav_feats = np.zeros(
            (self.buffer_size, self.max_uavs, self.d_uav), dtype=np.float32
        )
        self.uav_mask = np.zeros(
            (self.buffer_size, self.max_uavs), dtype=np.float32
        )
        self.sens_feats = np.zeros(
            (self.buffer_size, self.max_sensing_points, self.d_sens), dtype=np.float32
        )
        self.sens_mask = np.zeros(
            (self.buffer_size, self.max_sensing_points), dtype=np.float32
        )
        self.global_feats = np.zeros(
            (self.buffer_size, self.d_global), dtype=np.float32
        )
        
        # 动作 buffer
        self.uav_actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.sens_actions = np.zeros(self.buffer_size, dtype=np.int64)
        
        # 其他 buffer
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.uav_log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.sens_log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        
        # GAE 计算后的结果
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
    
    def add(
        self,
        obs: Dict[str, np.ndarray],
        uav_action: int,
        sens_action: int,
        reward: float,
        done: bool,
        value: float,
        uav_log_prob: float,
        sens_log_prob: float,
    ):
        """
        添加一个 macro step 的数据
        
        Args:
            obs: 观察 dict
            uav_action: 选择的 UAV ID
            sens_action: 选择的 sensing point ID (0 = SKIP)
            reward: 奖励
            done: 是否结束
            value: 价值估计
            uav_log_prob: UAV 选择的 log prob
            sens_log_prob: Sensing 选择的 log prob
        """
        self.uav_feats[self.pos] = obs['uav_feats']
        self.uav_mask[self.pos] = obs['uav_mask']
        self.sens_feats[self.pos] = obs['sens_feats']
        self.sens_mask[self.pos] = obs['sens_mask']
        self.global_feats[self.pos] = obs['global_feats']
        
        self.uav_actions[self.pos] = uav_action
        self.sens_actions[self.pos] = sens_action
        
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.uav_log_probs[self.pos] = uav_log_prob
        self.sens_log_probs[self.pos] = sens_log_prob
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantage(self, last_value: float, last_done: bool):
        """
        计算 GAE 和 returns
        
        Args:
            last_value: 最后一步的价值估计
            last_done: 最后一步是否结束
        """
        last_gae_lam = 0
        
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]  #TD error
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam 
            self.advantages[step] = last_gae_lam
        
        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]
    
    def get(self, batch_size: int) -> Generator[PreferenceRolloutBufferSamples, None, None]:
        """
        获取 minibatch 迭代器
        
        Args:
            batch_size: minibatch 大小
        
        Yields:
            PreferenceRolloutBufferSamples
        """
        indices = np.random.permutation(self.pos)
        
        for start in range(0, self.pos, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield self._get_samples(batch_indices)
    
    def _get_samples(self, indices: np.ndarray) -> PreferenceRolloutBufferSamples:
        """获取指定索引的样本"""
        return PreferenceRolloutBufferSamples(
            uav_feats=torch.FloatTensor(self.uav_feats[indices]).to(self.device),
            uav_mask=torch.FloatTensor(self.uav_mask[indices]).to(self.device),
            sens_feats=torch.FloatTensor(self.sens_feats[indices]).to(self.device),
            sens_mask=torch.FloatTensor(self.sens_mask[indices]).to(self.device),
            global_feats=torch.FloatTensor(self.global_feats[indices]).to(self.device),
            uav_actions=torch.LongTensor(self.uav_actions[indices]).to(self.device),
            sens_actions=torch.LongTensor(self.sens_actions[indices]).to(self.device),
            old_uav_log_probs=torch.FloatTensor(self.uav_log_probs[indices]).to(self.device),
            old_sens_log_probs=torch.FloatTensor(self.sens_log_probs[indices]).to(self.device),
            advantages=torch.FloatTensor(self.advantages[indices]).to(self.device),
            returns=torch.FloatTensor(self.returns[indices]).to(self.device),
            old_values=torch.FloatTensor(self.values[indices]).to(self.device),
        )
    
    @property
    def size(self) -> int:
        """当前 buffer 中的样本数量"""
        return self.pos


class PreferenceObsNormalizer:
    """
    观察归一化器
    
    对 Dict 格式的观察进行归一化。
    使用 running mean 和 std。
    """
    
    def __init__(self, max_uavs: int, max_sensing_points: int, d_uav: int = 9, d_sens: int = 8, d_global: int = 4):
        self.max_uavs = max_uavs
        self.max_sensing_points = max_sensing_points
        self.d_uav = d_uav
        self.d_sens = d_sens
        self.d_global = d_global
        
        # Running statistics
        self.uav_mean = np.zeros((max_uavs, d_uav), dtype=np.float32)
        self.uav_var = np.ones((max_uavs, d_uav), dtype=np.float32)
        self.sens_mean = np.zeros((max_sensing_points, d_sens), dtype=np.float32)
        self.sens_var = np.ones((max_sensing_points, d_sens), dtype=np.float32)
        self.global_mean = np.zeros(d_global, dtype=np.float32)
        self.global_var = np.ones(d_global, dtype=np.float32)
        
        self.count = 0
        self.epsilon = 1e-8
    
    def update(self, obs: Dict[str, np.ndarray]):
        """更新 running statistics"""
        self.count += 1
        
        # Welford's online algorithm
        delta_uav = obs['uav_feats'] - self.uav_mean
        self.uav_mean += delta_uav / self.count
        delta2_uav = obs['uav_feats'] - self.uav_mean
        self.uav_var += (delta_uav * delta2_uav - self.uav_var) / self.count
        
        delta_sens = obs['sens_feats'] - self.sens_mean
        self.sens_mean += delta_sens / self.count
        delta2_sens = obs['sens_feats'] - self.sens_mean
        self.sens_var += (delta_sens * delta2_sens - self.sens_var) / self.count
        
        delta_global = obs['global_feats'] - self.global_mean
        self.global_mean += delta_global / self.count
        delta2_global = obs['global_feats'] - self.global_mean
        self.global_var += (delta_global * delta2_global - self.global_var) / self.count
    
    def normalize(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """归一化观察"""
        return {
            'uav_feats': (obs['uav_feats'] - self.uav_mean) / np.sqrt(self.uav_var + self.epsilon),
            'uav_mask': obs['uav_mask'],
            'sens_feats': (obs['sens_feats'] - self.sens_mean) / np.sqrt(self.sens_var + self.epsilon),
            'sens_mask': obs['sens_mask'],
            'global_feats': (obs['global_feats'] - self.global_mean) / np.sqrt(self.global_var + self.epsilon),
        }

"""
Load PPO Policy

加载训练好的 PPO 模型，用于 Preference Stage 的速度决策。
提供统一的接口，支持不同的加载方式。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """初始化网络层"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOSpeedPolicy(nn.Module):
    """
    PPO 速度控制策略网络
    
    与 RPO_ContinusActionSpace.py 中的 UAVAgent 结构一致
    """
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        
        # Actor network for speed control
        self.actor_speed_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_speed_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic network (可选，用于评估)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
    
    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 观察 (batch, obs_dim)
            deterministic: 是否使用确定性策略
        
        Returns:
            speed_action: 速度比例 (batch, action_dim)，范围 [0, 1]
        """
        speed_mean = self.actor_speed_mean(x)
        
        if deterministic:
            # 确定性策略：直接使用 mean 并通过 sigmoid 映射到 [0, 1]
            speed_action = torch.sigmoid(speed_mean)
        else:
            # 随机策略：采样
            speed_logstd = self.actor_speed_logstd.expand_as(speed_mean)
            speed_std = torch.exp(speed_logstd)
            from torch.distributions import Normal
            speed_probs = Normal(speed_mean, speed_std)
            raw_action = speed_probs.sample()
            speed_action = torch.sigmoid(raw_action)
        
        return speed_action


class FrozenPPOPolicy:
    """
    冻结的 PPO 策略，用于 Preference Stage 的速度决策
    
    加载训练好的 RPO/PPO agent，设置为 eval 模式，
    提供简单的 act() 接口用于速度决策。
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        obs_dim: int = 9,
        action_dim: int = 1,
        device: str = "cpu"
    ):
        """
        Args:
            model_path: 模型文件路径（.pt 或 .pth）
                        如果为 None，则创建随机初始化的策略
            obs_dim: 观察空间维度
            action_dim: 动作空间维度（num_uavs）
            device: 设备 ("cpu" 或 "cuda")
        """
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 创建策略网络
        self.policy = PPOSpeedPolicy(obs_dim, action_dim).to(self.device)
        
        # 加载权重
        if model_path is not None and os.path.exists(model_path):
            self._load_weights(model_path)
            print(f"[FrozenPPOPolicy] Loaded weights from {model_path}")
        else:
            print(f"[FrozenPPOPolicy] Using randomly initialized policy")
        
        # 设置为 eval 模式并冻结参数
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad = False
    
    def _load_weights(self, model_path: str):
        """
        加载模型权重
        
        支持多种格式：
        1. 完整的 state_dict（只有网络参数）
        2. 包含 'model_state_dict' 的 checkpoint
        3. 包含 agent 完整状态的文件
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            # 尝试不同的 key
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'agent_state_dict' in checkpoint:
                state_dict = checkpoint['agent_state_dict']
            else:
                # 假设整个 checkpoint 就是 state_dict
                state_dict = checkpoint
            
            # 处理可能的 key 前缀差异
            # 例如 'actor_speed_mean.0.weight' vs 'policy.actor_speed_mean.0.weight'
            new_state_dict = {}
            for key, value in state_dict.items():
                # 移除可能的 'policy.' 前缀
                if key.startswith('policy.'):
                    new_key = key[7:]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            # 只加载匹配的 key
            policy_state = self.policy.state_dict()
            filtered_state = {
                k: v for k, v in new_state_dict.items()
                if k in policy_state and policy_state[k].shape == v.shape
            }
            
            if len(filtered_state) > 0:
                self.policy.load_state_dict(filtered_state, strict=False)
                print(f"[FrozenPPOPolicy] Loaded {len(filtered_state)} parameters")
            else:
                print(f"[FrozenPPOPolicy] Warning: No matching parameters found")
        else:
            print(f"[FrozenPPOPolicy] Warning: Unexpected checkpoint format")
    
    def act(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True
    ) -> np.ndarray:
        """
        输出速度动作
        
        Args:
            obs: 观察 (obs_dim,) 或 (batch, obs_dim)
            deterministic: 是否使用确定性策略
        
        Returns:
            speed_action: 速度比例 (action_dim,) 或 (batch, action_dim)，范围 [0, 1]
        """
        # 转换为 tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
        else:
            obs_tensor = obs.to(self.device)
        
        # 确保有 batch 维度
        single_input = obs_tensor.dim() == 1
        if single_input:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # 前向传播
        with torch.no_grad():
            speed_action = self.policy(obs_tensor, deterministic=deterministic)
        
        # 转换回 numpy
        speed_action = speed_action.cpu().numpy()
        
        # 移除 batch 维度（如果输入是单个样本）
        if single_input:
            speed_action = speed_action.squeeze(0)
        
        return speed_action
    
    def get_value(self, obs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        获取状态价值（可选，用于评估）
        
        Args:
            obs: 观察
        
        Returns:
            value: 状态价值
        """
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
        else:
            obs_tensor = obs.to(self.device)
        
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        with torch.no_grad():
            value = self.policy.critic(obs_tensor)
        
        return value.cpu().numpy().squeeze()


def load_frozen_policy(
    model_path: Optional[str] = None,
    num_uavs: int = 1,
    device: str = "cpu"
) -> FrozenPPOPolicy:
    """
    便捷函数：加载冻结的 PPO 策略
    
    Args:
        model_path: 模型路径
        num_uavs: UAV 数量
        device: 设备
    
    Returns:
        FrozenPPOPolicy 实例
    """
    obs_dim = 9 * num_uavs  # 每个 UAV 9 维状态
    action_dim = num_uavs   # 每个 UAV 1 维动作（速度比例）
    
    return FrozenPPOPolicy(
        model_path=model_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )


if __name__ == "__main__":
    # 测试
    print("Testing FrozenPPOPolicy...")
    
    # 创建随机初始化的策略
    policy = load_frozen_policy(model_path=None, num_uavs=1, device="cpu")
    
    # 测试单个样本
    obs = np.random.randn(9).astype(np.float32)
    action = policy.act(obs, deterministic=True)
    print(f"Single obs shape: {obs.shape}, action shape: {action.shape}")
    print(f"Action: {action}")
    
    # 测试 batch
    obs_batch = np.random.randn(4, 9).astype(np.float32)
    action_batch = policy.act(obs_batch, deterministic=True)
    print(f"Batch obs shape: {obs_batch.shape}, action shape: {action_batch.shape}")
    
    print("Test passed!")

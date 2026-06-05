"""
Macro Preference Environment

包装 UAVEnvironment + PreferenceEnvWrapper，
实现 Preference Stage 的 macro step：
1. 接收 (uav_id, sens_id) 动作
2. 应用 sensing 分配
3. 运行 K 个 low-level step（使用冻结的 PPO 决策速度）
4. 返回聚合奖励和下一个 macro 观察
"""

import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional, List, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.pref_env_interface import PreferenceEnvWrapper


class MacroPreferenceEnv:
    """
    Macro Preference Environment
    
    每个 macro step 包含 K 个 low-level step。
    Preference policy 选择 (UAV, sensing point) 配对，
    然后使用冻结的 PPO 执行 K 步速度决策。
    
    主要功能：
    - reset(): 重置环境，返回初始 macro 观察
    - step(pref_action): 执行 preference 动作，返回聚合奖励
    """
    
    def __init__(
        self,
        base_env,
        low_level_policy,
        pref_wrapper: Optional[PreferenceEnvWrapper] = None,
        K: int = 6,
        sensing_points: Optional[List[Dict]] = None,
        sensing_weight: float = 1.0,
        fixed_speed_value: float = 0.5,
    ):
        """
        Args:
            base_env: UAVEnvironment 实例
            low_level_policy: 冻结的 PPO 策略（用于速度决策），None 时使用固定速度
            pref_wrapper: PreferenceEnvWrapper 实例（如果为 None，自动创建）
            K: 每个 macro step 的 low-level step 数量
            sensing_points: 感知点配置列表
            sensing_weight: 感知超时惩罚权重（宏观层）；订单奖励权重 order_weight 由下层 UAVEnvironment 的 reward_config 读取
            fixed_speed_value: 当 low_level_policy=None 时使用的固定速度比例 [0,1]
        """
        self.base_env = base_env
        self.low_policy = low_level_policy
        self.K = K
        self.sensing_weight = sensing_weight
        self.fixed_speed_value = fixed_speed_value
        
        # 创建或使用 PreferenceEnvWrapper
        if pref_wrapper is not None:
            self.pref_wrapper = pref_wrapper
        else:
            self.pref_wrapper = PreferenceEnvWrapper(
                    base_env=base_env,
                    sensing_points=sensing_points,
                )
        
        # 统计信息
        self.total_low_steps = 0
        self.total_macro_steps = 0
        self.episode_sensing_completed = 0
        self.episode_orders_completed = 0
    
    def reset(self, seed: Optional[int] = None, sensing_points: Optional[List[Dict]] = None) -> Dict[str, np.ndarray]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            sensing_points: 新的感知点配置（可选）
        
        Returns:
            macro_obs: macro 观察
        """
        # 重置 base 环境
        obs, info = self.base_env.reset(seed=seed)
        
        # 重置 sensing 状态
        self.pref_wrapper.reset_sensing_points(sensing_points)
        
        # 重置统计
        self.total_low_steps = 0
        self.total_macro_steps = 0
        self.episode_sensing_completed = 0
        self.episode_orders_completed = 0
        
        return self._build_macro_obs()
    
    def step(self, pref_action: Tuple[int, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        执行一个 macro step
        
        Args:
            pref_action: (uav_id, sens_id) 
                - uav_id: 选择的 UAV ID
                - sens_id: 选择的 sensing point ID (0 = SKIP)
        
        Returns:
            macro_obs: 下一个 macro 观察
            total_reward: 聚合奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 附加信息
        """
        uav_id, sens_id = pref_action
        
        # 1. 应用 preference 决策（分配 sensing 任务）
        assignment_success = self.pref_wrapper.apply_sensing_assignment(uav_id, sens_id)
        
        # 2. 运行 K 个 low-level step
        total_sensing_entropy_reward = 0.0        # sensing 熵奖励（已×sensing_weight）
        total_sensing_timeout_penalty = 0.0       # sensing 超时惩罚（原始值，未加权）
        total_delivery_reward = 0.0               # 订单奖励：完成收益+超时惩罚（含 order_weight）
        total_delivery_completion_reward = 0.0    # 纯配送完成收益（>= 0），业务 objective 用
        total_order_timeout_penalty = 0.0         # 纯订单超时惩罚（<= 0），辅助指标
        total_energy_penalty = 0.0                # 纯能耗惩罚（负数）
        total_shaping_reward = 0.0                # shaping 辅助奖励
        terminated = False
        truncated = False
        info = {}
        low_step_snapshots = []
        
        for k in range(self.K):
            # 获取 low-level 观察
            low_obs = self.pref_wrapper.get_low_obs()
            
            # 使用冻结的 PPO 选择速度
            speed_action = self._low_policy_act(low_obs)
            
            # 执行 low-level step
            # reward 包含：订单奖励 + sensing熵奖励（×sensing_weight）+ 能量惩罚 + shaping
            next_obs, reward, terminated, truncated, step_info = self.base_env.step(speed_action)
            info = step_info
            low_step_snapshots.append({
                'macro_inner_step': k,
                'global_low_step': self.total_low_steps + 1,
                'current_time': step_info.get('current_time'),
                'uav_positions': step_info.get('uav_positions', []),
                'uav_batteries': step_info.get('uav_batteries', []),
                'uav_task_status': step_info.get('uav_task_status', []),
                # 每个 UAV 的逐步物理量（case study 用）
                'uav_speed_ratio': step_info.get('uav_speed_ratio', []),
                'uav_actual_speed': step_info.get('uav_actual_speed', []),
                'uav_effective_speed': step_info.get('uav_effective_speed', []),
                'uav_battery_consumption': step_info.get('uav_battery_consumption', []),
                'uav_energy_penalty_per_uav': step_info.get('uav_energy_penalty_per_uav', []),
            })
            
            # 检查 sensing 超时和更新完成状态
            timeout_penalty, completed_ids = self.pref_wrapper.check_sensing_completion(step_info)
            total_sensing_timeout_penalty += timeout_penalty
            total_sensing_entropy_reward += step_info.get('step_sensing_entropy_reward', 0.0)  # 已×sensing_weight
            self.episode_sensing_completed += len(completed_ids)
            
            # 从 step_info 直接读取各分量（避免用减法倒推导致命名混乱）
            total_delivery_reward            += step_info.get('step_delivery_reward', 0.0)
            total_delivery_completion_reward += step_info.get('step_delivery_completion_reward', 0.0)
            total_order_timeout_penalty      += step_info.get('step_order_timeout_penalty', 0.0)
            total_energy_penalty             += step_info.get('step_energy_penalty', 0.0)
            total_shaping_reward             += step_info.get('step_shaping_reward', 0.0)
            
            self.total_low_steps += 1
            
            if terminated or truncated:
                break
        
        # sensing_reward = 熵函数奖励（已×sensing_weight）+ 超时惩罚（×sensing_weight）
        total_sensing_reward = total_sensing_entropy_reward + self.sensing_weight * total_sensing_timeout_penalty
        
        total_reward = (
            total_sensing_reward
            + total_delivery_reward
            + total_energy_penalty
            + total_shaping_reward
        )
        
        self.total_macro_steps += 1
        
        # 4. 更新订单完成统计
        if 'total_orders_completed' in info:
            self.episode_orders_completed = info['total_orders_completed']
        
        # 5. 构建 macro 观察
        macro_obs = self._build_macro_obs()
        
        # 6. 构建 info（移除旧的有歧义 order_reward，补充各分量）
        sensing_stats = self.pref_wrapper.get_sensing_stats()
        info.update({
            
            'macro_step': self.total_macro_steps,
            'low_steps': self.total_low_steps,
            'sensing_completed': self.episode_sensing_completed,
            'orders_completed': self.episode_orders_completed,
            # ---- 奖励分量（语义清晰） ----
            'delivery_reward': total_delivery_reward,                      # 订单：完成+超时（训练用）
            'delivery_completion_reward': total_delivery_completion_reward,  # 纯配送完成收益（>= 0），业务 objective 用
            'order_timeout_penalty': total_order_timeout_penalty,          # 纯订单超时惩罚（<= 0），辅助指标
            'energy_penalty': total_energy_penalty,                        # 能耗（负数）
            'shaping_reward': total_shaping_reward,                        # shaping 辅助
            'sensing_reward': total_sensing_reward,                        # 感知总奖励（熵+超时加权，训练用）
            'sensing_entropy_reward': total_sensing_entropy_reward,        # 纯 sensing 熵奖励（已×weight），业务 objective 用
            'sensing_timeout_penalty': total_sensing_timeout_penalty,      # sensing 超时惩罚（原始），辅助指标
            # --------------------------------
            'assignment_success': assignment_success,
            'sensing_stats': sensing_stats,
            'low_step_snapshots': low_step_snapshots,
        })
        
        # # Macro step 汇总打印（替代原来逐步打印，信息更聚焦）
        # print(
        #     f"[Macro] step={self.total_macro_steps}, action=({uav_id},{sens_id}), assign={assignment_success} | "
        #     f"delivery={total_delivery_reward:.2f}, "
        #     f"sensing={total_sensing_reward:.2f}(entropy={total_sensing_entropy_reward:.2f}, timeout_raw={total_sensing_timeout_penalty:.4f}), "
        #     f"energy={total_energy_penalty:.4f}, shaping={total_shaping_reward:.4f}(not in reward) | "
        #     f"total_reward={float(total_reward):.4f} | "
        #     f"sensing_stats={sensing_stats}"
        # )
        
        return macro_obs, float(total_reward), terminated, truncated, info
    
    def _low_policy_act(self, low_obs: np.ndarray) -> np.ndarray:
        """
        使用冻结的 PPO 策略选择速度
        
        Args:
            low_obs: low-level 观察
        
        Returns:
            speed_action: 速度动作
        """
        if self.low_policy is not None:
            return self.low_policy.act(low_obs, deterministic=True)
        else:
            num_uavs = self.base_env.num_uavs
            return np.full(num_uavs, self.fixed_speed_value, dtype=np.float32)
    
    def _build_macro_obs(self) -> Dict[str, np.ndarray]:
        """
        构建 macro 观察
        
        直接使用 PreferenceEnvWrapper 的 get_pref_obs()
        
        Returns:
            macro_obs: Dict 格式的观察
        """
        return self.pref_wrapper.get_pref_obs()
    
    def get_sensing_target_for_uav(self, uav_id: int):
        """获取 UAV 的 sensing 目标"""
        return self.pref_wrapper.get_sensing_target_for_uav(uav_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_macro_steps': self.total_macro_steps,
            'total_low_steps': self.total_low_steps,
            'episode_sensing_completed': self.episode_sensing_completed,
            'episode_orders_completed': self.episode_orders_completed,
            'sensing_stats': self.pref_wrapper.get_sensing_stats(),
        }
    
    @property
    def num_uavs(self) -> int:
        """UAV 数量"""
        return self.pref_wrapper.num_uavs
    
    @property
    def max_sensing_points(self) -> int:
        """最大 sensing 点数量"""
        return self.pref_wrapper.max_sensing_points
    
    def close(self):
        """关闭环境"""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()


def create_macro_env(
    base_env,
    low_level_policy=None,
    sensing_points: Optional[List[Dict]] = None,
    K: int = 6,
    sensing_weight: float = 1.0,
    max_sensing_points: int = 50,
    sensing_completion_distance: float = 100.0,
    sensing_reward_base: float = 10.0,
    sensing_timeout_penalty: float = -5.0,
    fixed_speed_value: float = 0.5,
) -> MacroPreferenceEnv:
    """
    便捷函数：创建 MacroPreferenceEnv

    Args:
        base_env: UAVEnvironment
        low_level_policy: 冻结的 PPO 策略（None 时使用固定速度）
        sensing_points: 感知点配置
        K: macro step 包含的 low-level step 数量
        sensing_weight: sensing 超时惩罚宏观放大系数（订单权重 order_weight 由 UAVEnvironment.reward_config 管理）
        max_sensing_points: 最大感知点数量
        sensing_completion_distance: 完成感知的距离阈值（米）
        sensing_reward_base: 感知基础奖励
        sensing_timeout_penalty: 感知超时惩罚（原始值，未乘 sensing_weight）
        fixed_speed_value: 当 low_level_policy=None 时使用的固定速度比例 [0,1]

    Returns:
        MacroPreferenceEnv
    """
    pref_wrapper = PreferenceEnvWrapper(
        base_env=base_env,
        sensing_points=sensing_points,
        max_sensing_points=max_sensing_points,
        sensing_completion_distance=sensing_completion_distance,
        sensing_reward_base=sensing_reward_base,
        sensing_timeout_penalty=sensing_timeout_penalty,
    )

    return MacroPreferenceEnv(
        base_env=base_env,
        low_level_policy=low_level_policy,
        pref_wrapper=pref_wrapper,
        K=K,
        sensing_weight=sensing_weight,
        fixed_speed_value=fixed_speed_value,
    )

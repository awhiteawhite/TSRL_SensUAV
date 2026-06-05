"""
Preference Environment Interface

为 UAVEnvironment 提供 Preference Stage 所需的接口。
这是一个独立的 Wrapper 类，不修改 UAVEnvironment 的现有代码。

使用方法：
    base_env = UAVEnvironment(...)
    pref_env = PreferenceEnvWrapper(base_env, sensing_points_config)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SensingPoint:
    """
    感知点数据结构
    
    Attributes:
        point_id: 感知点 ID
        location: (lat, lng) 位置
        priority: 优先级（0-1）
        time_window: (start_time, end_time) 时间窗
        data_value: 数据价值（用于计算奖励）
        is_active: 是否激活（未被完成）
    """
    point_id: int
    location: Tuple[float, float]
    priority: float
    time_window: Tuple[datetime, datetime]
    data_value: float = 1.0
    is_active: bool = True
    assigned_uav_id: Optional[int] = None  # 被分配的 UAV ID


class PreferenceEnvWrapper:
    """
    Preference Stage 环境包装器
    
    包装 UAVEnvironment，提供 Preference Stage 所需的接口：
    - get_pref_obs(): 获取 preference 观察
    - apply_sensing_assignment(): 应用 sensing 分配
    - get_low_obs(): 获取 low-level PPO 观察
    
    不修改原始 UAVEnvironment 的代码，通过包装方式扩展功能。
    """
    
    def __init__(
        self,
        base_env,
        sensing_points: Optional[List[Dict]] = None,
        max_sensing_points: int = 50,
        sensing_completion_distance: float = 100.0,  # 米
        sensing_reward_base: float = 10.0,
        sensing_timeout_penalty: float = -5.0,
    ):
        """
        Args:
            base_env: UAVEnvironment 实例
            sensing_points: 感知点配置列表（如果为 None，则从环境配置加载）
            max_sensing_points: 最大感知点数量（用于 padding）
            sensing_completion_distance: 完成感知的距离阈值（米）
            sensing_reward_base: 完成感知的基础奖励
            sensing_timeout_penalty: 感知超时惩罚
        """
        self.base_env = base_env
        self.max_sensing_points = max_sensing_points
        self.sensing_completion_distance = sensing_completion_distance
        self.sensing_reward_base = sensing_reward_base
        self.sensing_timeout_penalty = sensing_timeout_penalty
        
        # 从 base_env 获取配置
        self.num_uavs = getattr(base_env, 'num_uavs', 1)
        self.depot_location = getattr(base_env, 'depot_location', (0.0, 0.0))
        
        # 感知点管理
        self.sensing_points: List[SensingPoint] = []
        self.completed_sensing_ids: List[int] = []
        self.timeout_sensing_ids: List[int] = []
        
        # UAV 当前分配的 sensing 目标
        self.uav_sensing_targets: Dict[int, Optional[int]] = {
            i: None for i in range(self.num_uavs)
        }
        
        # 初始化感知点
        if sensing_points is not None:
            self._init_sensing_points(sensing_points)
    
    def _init_sensing_points(self, sensing_configs: List[Dict]):
        """
        从配置初始化感知点？？这里是不是要改成从数据库中读取感知点？？
        
        Args:
            sensing_configs: [
                {
                    'point_id': 0,
                    'location': (lat, lng),
                    'priority': 0.8,
                    'time_window_minutes': 60,  # 从当前时间开始的时间窗长度
                    'data_value': 1.0
                },
                ...
            ]
        """
        current_time = self.base_env.current_time  # 使用仿真时间
        
        self.sensing_points = []
        for cfg in sensing_configs:
            # 计算时间窗
            window_minutes = cfg.get('time_window_minutes', 60)
            start_time = current_time
            end_time = current_time + timedelta(minutes=window_minutes)
            
            point = SensingPoint(
                point_id=cfg['point_id'],
                location=tuple(cfg['location']),
                priority=cfg.get('priority', 1.0),
                time_window=(start_time, end_time),
                data_value=cfg.get('data_value', 1.0),
                is_active=True,
                assigned_uav_id=None
            )
            self.sensing_points.append(point)
    
    def reset_sensing_points(self, sensing_configs: Optional[List[Dict]] = None):
        """
        重置感知点状态（用于新 episode）
        
        Args:
            sensing_configs: 新的感知点配置（如果为 None，则使用现有配置重置状态，
                并基于当前 episode 的 current_time 更新 time_window，避免跨 episode 时间错位导致批量超时）
        """
        if sensing_configs is not None:
            self._init_sensing_points(sensing_configs)
        else:
            # 重置所有感知点状态，并用当前 episode 的 current_time 更新 time_window
            current_time = self.base_env.current_time
            for point in self.sensing_points:
                point.is_active = True
                point.assigned_uav_id = None
                # 保持 duration 不变，将 time_window 平移到当前 episode
                duration = point.time_window[1] - point.time_window[0]
                point.time_window = (current_time, current_time + duration)
        
        self.completed_sensing_ids = []
        self.timeout_sensing_ids = []
        self.uav_sensing_targets = {i: None for i in range(self.num_uavs)}
    
    def get_pref_obs(self) -> Dict[str, np.ndarray]:
        """
        提取 Preference Stage 的观察
        
        Returns:
            {
                'uav_feats': (N, D_uav),      # 每个 UAV 的特征
                'uav_mask': (N,),             # UAV 有效性 mask
                'sens_feats': (M, D_sens),    # 每个 sensing point 的特征
                'sens_mask': (M,),            # Sensing point 有效性 mask
                'global_feats': (D_global,)   # 全局特征
            }
        """
        # ============ UAV Features ============
        # 特征维度 D_uav = 9:
        # [lat, lng, battery, wind_speed, wind_direction, precipitation, 
        #  has_order, task_status, is_available]
        uav_feats = np.zeros((self.num_uavs, 9), dtype=np.float32)
        uav_mask = np.ones(self.num_uavs, dtype=np.float32)
        
        for uav_id in range(self.num_uavs):
            # 从 base_env 获取 UAV 状态
            if hasattr(self.base_env, 'uav_positions') and len(self.base_env.uav_positions) > uav_id:
                lat, lng = self.base_env.uav_positions[uav_id]
            else:
                lat, lng = self.depot_location
            
            battery = self.base_env.uav_batteries[uav_id] if hasattr(self.base_env, 'uav_batteries') else 1.0
            
            # 天气信息（从 state 中提取或使用默认值）--修复逻辑，之后要把 if/else结构去掉以防报错
            if hasattr(self.base_env, 'uav_states') and len(self.base_env.uav_states) > uav_id:
                state = self.base_env.uav_states[uav_id]
                wind_speed = state[3] if len(state) > 3 else 0.0
                wind_direction = state[4] if len(state) > 4 else 0.0
                precipitation = state[5] if len(state) > 5 else 0.0
            else:
                wind_speed, wind_direction, precipitation = 0.0, 0.0, 0.0
            
            # 任务状态
            has_order = 0.0
            task_status = 0.0
            is_available = 1.0
            
            if hasattr(self.base_env, 'uav_task_managers') and len(self.base_env.uav_task_managers) > uav_id:
                task_manager = self.base_env.uav_task_managers[uav_id]
                has_order = 1.0 if task_manager.current_order is not None else 0.0
                # 任务状态编码：0=idle, 1=assigned, 2=delivery, 3=returning, 4=charging, 5=low_battery
                from environments.order_system import UAVTaskStatus
                status_map = {
                    UAVTaskStatus.IDLE: 0.0,
                    UAVTaskStatus.ASSIGNED: 1.0,
                    UAVTaskStatus.DELIVERY: 2.0,
                    UAVTaskStatus.RETURNING: 3.0,
                    UAVTaskStatus.CHARGING: 4.0,
                    UAVTaskStatus.LOW_BATTERY_RETURN: 5.0,
                }
                task_status = status_map.get(task_manager.task_status, 0.0)
                # 使用 is_available_for_sensing() 判断是否可接受 sensing 任务
                is_available = 1.0 if task_manager.is_available_for_sensing() else 0.0
            
            uav_feats[uav_id] = [
                lat, lng, battery, wind_speed, wind_direction, precipitation,
                has_order, task_status, is_available
            ]
        
        # ============ Sensing Point Features ============
        # 特征维度 D_sens = 8:
        # [lat, lng, priority, time_remaining_ratio, data_value, 
        #  distance_to_depot, is_assigned, assigned_uav_id]
        sens_feats = np.zeros((self.max_sensing_points, 8), dtype=np.float32)
        sens_mask = np.zeros(self.max_sensing_points, dtype=np.float32)
        
        current_time = self.base_env.current_time  # 使用仿真时间
        
        active_points = [p for p in self.sensing_points if p.is_active]
        for i, point in enumerate(active_points[:self.max_sensing_points]):
            lat, lng = point.location
            
            # 计算时间窗剩余比例
            total_window = (point.time_window[1] - point.time_window[0]).total_seconds()
            remaining = max(0, (point.time_window[1] - current_time).total_seconds())
            time_remaining_ratio = remaining / max(total_window, 1.0)
            
            # 计算到仓库的距离（归一化）
            depot_lat, depot_lng = self.depot_location
            distance_to_depot = np.sqrt((lat - depot_lat)**2 + (lng - depot_lng)**2)
            # 假设最大距离为区域对角线
            max_dist = 0.1  # 约 10km 的经纬度差
            distance_to_depot_norm = min(distance_to_depot / max_dist, 1.0)
            
            is_assigned = 1.0 if point.assigned_uav_id is not None else 0.0
            assigned_uav = float(point.assigned_uav_id) if point.assigned_uav_id is not None else -1.0
            
            sens_feats[i] = [
                lat, lng, point.priority, time_remaining_ratio, point.data_value,
                distance_to_depot_norm, is_assigned, assigned_uav
            ]
            sens_mask[i] = 1.0
        
        # ============ Global Features ============
        # 特征维度 D_global = 4:
        # [time_of_day_norm, pending_orders_ratio, active_sensing_ratio, avg_battery]
        
        # 时间归一化（0-1，一天中的位置）
        time_of_day_norm = (current_time.hour * 60 + current_time.minute) / (24 * 60)
        
        # 待处理订单比例
        pending_orders = len(getattr(self.base_env, 'pending_orders', []))
        max_pending = getattr(self.base_env, 'max_pending_orders', 50)
        pending_orders_ratio = pending_orders / max(max_pending, 1)
        
        # 活跃感知点比例
        active_sensing_ratio = len(active_points) / max(self.max_sensing_points, 1)
        
        # 平均电量
        avg_battery = np.mean([
            self.base_env.uav_batteries[i] if hasattr(self.base_env, 'uav_batteries') else 1.0
            for i in range(self.num_uavs)
        ])
        
        global_feats = np.array([
            time_of_day_norm, pending_orders_ratio, active_sensing_ratio, avg_battery
        ], dtype=np.float32)
        
        return {
            'uav_feats': uav_feats,
            'uav_mask': uav_mask,
            'sens_feats': sens_feats,
            'sens_mask': sens_mask,
            'global_feats': global_feats
        }
    
    def apply_sensing_assignment(self, uav_id: int, sens_id: int) -> bool:
        """
        应用 Preference Stage 的决策：将 sensing point 分配给 UAV
        
        同时更新：
        1. PreferenceEnvWrapper 内部的 sensing_points 状态
        2. UAVTaskManager 的 sensing_target（用于 UAV 移动决策）
        
        Args:
            uav_id: UAV 索引
            sens_id: Sensing point 索引（0 表示 SKIP，不分配）
        
        Returns:
            success: 是否成功分配
        """
        # sens_id = 0 表示 SKIP
        if sens_id == 0:
            # 清除该 UAV 的 sensing 目标
            old_target = self.uav_sensing_targets.get(uav_id)
            if old_target is not None:
                # 释放之前分配的 sensing point
                for point in self.sensing_points:
                    if point.point_id == old_target:
                        point.assigned_uav_id = None
                        break
            self.uav_sensing_targets[uav_id] = None
            
            # 同步到 UAVTaskManager：取消 sensing 任务
            if hasattr(self.base_env, 'uav_task_managers') and len(self.base_env.uav_task_managers) > uav_id:
                self.base_env.uav_task_managers[uav_id].cancel_sensing_task()
            
            return True
        
        # sens_id 是 1-indexed（因为 0 是 SKIP）
        actual_sens_idx = sens_id - 1
        
        # 获取活跃感知点
        active_points = [p for p in self.sensing_points if p.is_active]
        
        if actual_sens_idx < 0 or actual_sens_idx >= len(active_points):
            return False
        
        target_point = active_points[actual_sens_idx]
        
        # 检查是否已被分配给其他 UAV
        if target_point.assigned_uav_id is not None and target_point.assigned_uav_id != uav_id:
            return False
        
        # 检查 UAV 是否可接受 sensing 任务
        if hasattr(self.base_env, 'uav_task_managers') and len(self.base_env.uav_task_managers) > uav_id:
            task_manager = self.base_env.uav_task_managers[uav_id]
            if not task_manager.is_available_for_sensing():
                return False
        
        # 清除该 UAV 之前的分配
        old_target = self.uav_sensing_targets.get(uav_id)
        if old_target is not None and old_target != target_point.point_id:
            for point in self.sensing_points:
                if point.point_id == old_target:
                    point.assigned_uav_id = None
                    break
        
        # 执行分配（PreferenceEnvWrapper 内部状态）
        target_point.assigned_uav_id = uav_id
        self.uav_sensing_targets[uav_id] = target_point.point_id
        
        # 同步到 UAVTaskManager（用于 UAV 移动决策）
        if hasattr(self.base_env, 'uav_task_managers') and len(self.base_env.uav_task_managers) > uav_id:
            self.base_env.uav_task_managers[uav_id].assign_sensing_task(
                location=target_point.location,
                point_id=target_point.point_id
            )
        
        return True
    
    def get_sensing_target_for_uav(self, uav_id: int) -> Optional[Tuple[float, float]]:
        """
        获取 UAV 被分配的 sensing 目标位置
        
        Args:
            uav_id: UAV ID
        
        Returns:
            location: (lat, lng) 或 None（如果没有分配）
        """
        target_id = self.uav_sensing_targets.get(uav_id)
        if target_id is None:
            return None
        
        for point in self.sensing_points:
            if point.point_id == target_id and point.is_active:
                return point.location
        
        return None
    
    def check_sensing_completion(self, step_info: Optional[Dict] = None) -> Tuple[float, List[int]]:
        """
        检查感知完成情况
        
        职责分工：
        - sensing 完成检测和奖励计算：由 uav_environment.step() 负责
        - 本方法负责：
          1. 检查超时（并同步到 task_manager）
          2. 根据 step_info 更新内部 sensing_points 状态
        
        Args:
            step_info: uav_environment.step() 返回的 info 字典，包含 'sensing_completed_ids'
        
        Returns:
            timeout_penalty: 本步的超时惩罚（完成奖励由 uav_environment 计算）
            completed_ids: 本步完成的感知点 ID 列表
        """
        timeout_penalty = 0.0
        completed_ids = []
        current_time = self.base_env.current_time  # 直接使用仿真时间
        
        # ========== 1. 检查超时 ==========
        for point in self.sensing_points:
            if not point.is_active:
                continue
            
            # 检查超时
            if current_time > point.time_window[1]:
                point.is_active = False
                self.timeout_sensing_ids.append(point.point_id)
                
                # 超时惩罚
                timeout_penalty += self.sensing_timeout_penalty * point.data_value
                
                # 清除 PreferenceEnvWrapper 内部分配
                if point.assigned_uav_id is not None:
                    uav_id = point.assigned_uav_id
                    self.uav_sensing_targets[uav_id] = None
                    point.assigned_uav_id = None
                    
                    # 同步到 UAVTaskManager：取消 sensing 任务
                    if hasattr(self.base_env, 'uav_task_managers') and len(self.base_env.uav_task_managers) > uav_id:
                        self.base_env.uav_task_managers[uav_id].cancel_sensing_task()
        
        # ========== 2. 根据 step_info 更新完成状态 ==========
        if step_info is not None:
            sensing_completed_ids = step_info.get('sensing_completed_ids', [])
            
            for completed_id in sensing_completed_ids:
                # 更新 PreferenceEnvWrapper 内部状态
                for point in self.sensing_points:
                    if point.point_id == completed_id and point.is_active:
                        point.is_active = False
                        self.completed_sensing_ids.append(completed_id)
                        completed_ids.append(completed_id)
                        
                        # 清除分配记录
                        if point.assigned_uav_id is not None:
                            self.uav_sensing_targets[point.assigned_uav_id] = None
                            point.assigned_uav_id = None
                        break
        
        # 注意：sensing 完成的奖励由 uav_environment 通过 sensing_reward_system 计算
        # 这里返回的仅包含超时惩罚
        
        return timeout_penalty, completed_ids
    
    def get_low_obs(self) -> np.ndarray:
        """
        获取 low-level PPO 的观察（用于速度决策）
        
        直接返回 base_env 的 state
        
        Returns:
            state: (obs_dim,) 与基础 PPO 训练时一致的观察格式
        """
        if hasattr(self.base_env, 'state') and self.base_env.state is not None:
            return self.base_env.state.copy()
        else:
            # 返回默认观察
            return np.zeros(9 * self.num_uavs, dtype=np.float32)
    
    def get_sensing_stats(self) -> Dict[str, Any]:
        """
        获取感知统计信息
        
        Returns:
            stats: {
                'total_sensing_points': int,
                'active_sensing_points': int,
                'completed_sensing_points': int,
                'timeout_sensing_points': int,
                'assigned_sensing_points': int
            }
        """
        active = [p for p in self.sensing_points if p.is_active]
        assigned = [p for p in active if p.assigned_uav_id is not None]
        
        return {
            'total_sensing_points': len(self.sensing_points),
            'active_sensing_points': len(active),
            'completed_sensing_points': len(self.completed_sensing_ids),
            'timeout_sensing_points': len(self.timeout_sensing_ids),
            'assigned_sensing_points': len(assigned)
        }

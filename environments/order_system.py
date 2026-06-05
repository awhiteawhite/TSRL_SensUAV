"""
订单系统模块
包含订单类和订单管理逻辑
"""

import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Tuple, List
import random


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"      # 等待分配UAV
    ASSIGNED = "assigned"    # 已分配给UAV
    IN_PROGRESS = "in_progress"  # 正在执行
    COMPLETED = "completed"  # 已完成
    TIMEOUT = "timeout"      # 超时


class Order:
    """
    订单类
    表示一个配送订单，包含起点、终点、时间窗等信息
    
    注意：时间窗从UAV取到订单的时间（start_time）开始计算，
    在调用start_delivery()时会重新计算并更新时间窗。
    """

    def __init__(self,
                 order_id: str,
                 start_location: Tuple[float, float],  # (lat, lng)
                 end_location: Tuple[float, float],    # (lat, lng)
                 time_window: Tuple[datetime, datetime],  # (start_time, end_time) 占位符，会在start_delivery时更新
                 accept_time: datetime,
                 delivery_time: Optional[datetime] = None):
        """
        初始化订单

        Args:
            order_id: 订单ID
            start_location: 起点坐标 (lat, lng)
            end_location: 终点坐标 (lat, lng)
            time_window: 时间窗占位符 (start_time, end_time)，会在start_delivery时基于pickup_time重新计算
            accept_time: 订单接受时间
            delivery_time: 实际配送完成时间（可选）
        """
        self.order_id = order_id
        self.start_location = start_location
        self.end_location = end_location
        self.time_window = time_window
        self.accept_time = accept_time
        self.delivery_time = delivery_time

        # 订单状态
        self.status = OrderStatus.PENDING
        self.assigned_uav_id = None  # 分配的UAV ID

        # 时间相关
        self.assignment_time = None  # 分配给UAV的时间
        self.start_time = None       # 开始执行时间
        self.completion_time = None  # 完成时间

    @property
    def time_window_start(self) -> datetime:
        """
        时间窗开始时间
        
        如果订单已开始（start_time不为None），返回基于start_time计算的时间窗起点
        否则返回初始化时的时间窗起点（占位符）
        """
        if self.time_window is not None:
            return self.time_window[0]
        # 如果时间窗还未计算，返回None或抛出异常
        raise ValueError("Time window not yet calculated. Order must be started first.")

    @property
    def time_window_end(self) -> datetime:
        """
        时间窗结束时间
        
        如果订单已开始（start_time不为None），返回基于start_time计算的时间窗终点
        否则返回初始化时的时间窗终点（占位符）
        """
        if self.time_window is not None:
            return self.time_window[1]
        # 如果时间窗还未计算，返回None或抛出异常
        raise ValueError("Time window not yet calculated. Order must be started first.")

    def is_overdue(self, current_time: datetime) -> bool:
        """
        检查是否超时
        
        时间窗从start_time（UAV取到订单的时间）开始计算
        
        Args:
            current_time: 当前仿真时间（不是系统真实时间）
        """
        if self.start_time is None:
            # 如果订单还没开始，返回False
            return False
        return current_time > self.time_window_end and self.status != OrderStatus.COMPLETED

    @property
    def is_within_time_window(self) -> bool:
        """
        检查是否在时间窗内完成配送
        
        时间窗从start_time（UAV取到订单的时间）开始计算
        """
        if self.completion_time is None or self.start_time is None:
            return False
        # 时间窗已经在start_delivery时基于start_time计算并更新
        return self.time_window_start <= self.completion_time <= self.time_window_end

    @property
    def delivery_distance(self) -> float:
        """计算配送距离"""
        lat1, lng1 = self.start_location
        lat2, lng2 = self.end_location
        return np.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)

    def assign_to_uav(self, uav_id: int, assignment_time: datetime):
        """分配订单给指定的UAV"""
        self.assigned_uav_id = uav_id
        self.assignment_time = assignment_time
        self.status = OrderStatus.ASSIGNED

    def start_delivery(self, start_time: datetime, min_window: int = 30, max_window: int = 50):
        """
        开始配送（UAV取到订单）
        
        Args:
            start_time: UAV取到订单的时间（即时间窗起点）
            min_window: 最小时间窗长度（分钟）
            max_window: 最大时间窗长度（分钟）
        """
        self.start_time = start_time
        self.status = OrderStatus.IN_PROGRESS
        
        # 从start_time开始计算时间窗
        self.time_window = generate_order_time_window(
            pickup_time=start_time,
            min_window=min_window,
            max_window=max_window
        )

    def complete_delivery(self, completion_time: datetime):
        """完成配送"""
        self.completion_time = completion_time
        self.status = OrderStatus.COMPLETED

    def mark_timeout(self):
        """标记为超时"""
        self.status = OrderStatus.TIMEOUT

    def __str__(self):
        return (".2f"
                ".2f"
                f"{self.status.value}")


class UAVTaskStatus(Enum):
    """UAV任务状态枚举"""
    IDLE = "idle"                    # 空闲，在起点等待
    ASSIGNED = "assigned"           # 已分配订单，前往订单起点
    DELIVERY = "delivery"           # 执行配送任务（已到达起点，前往订单终点）
    RETURNING = "returning"         # 返回起点
    CHARGING = "charging"           # 充电中
    LOW_BATTERY_RETURN = "low_battery_return"  # 电量不足强制返回


class UAVTaskManager:
    """
    UAV任务管理器
    管理单个UAV的任务状态和订单分配
    
    支持双层任务：
    - sensing task（高优先级，可选）：Preference Stage 分配的感知任务
    - order task（原有系统）：订单配送任务
    
    目标优先级：sensing_target > order_target（但 LOW_BATTERY 和 CHARGING 在 step 中单独处理）
    """

    def __init__(self, uav_id: int, depot_location: Tuple[float, float]):
        """
        初始化UAV任务管理器

        Args:
            uav_id: UAV ID
            depot_location: 仓库位置 (lat, lng)
        """
        self.uav_id = uav_id
        self.depot_location = depot_location

        # 任务状态
        self.task_status = UAVTaskStatus.IDLE
        self.current_order = None  # 当前执行的订单

        # 位置相关
        self.target_location = None  # 当前目标位置（订单相关）

        # 时间跟踪
        self.last_status_change = None
        
        # ========== Sensing Task 相关字段（Preference Stage 使用） ==========
        self.sensing_target: Optional[Tuple[float, float]] = None  # sensing 位置 (lat, lng)
        self.sensing_point_id: Optional[int] = None  # sensing 点 ID（用于完成时回调）
    
    # ========== Sensing Task 相关方法 ==========
    
    def assign_sensing_task(self, location: Tuple[float, float], point_id: int) -> bool:
        """
        分配 sensing 任务
        
        Args:
            location: sensing 点位置 (lat, lng)
            point_id: sensing 点 ID
        
        Returns:
            是否成功分配
        """
        if not self.is_available_for_sensing():
            return False
        
        self.sensing_target = location
        self.sensing_point_id = point_id
        return True
    
    def complete_sensing_task(self) -> Optional[int]:
        """
        完成 sensing 任务
        
        Returns:
            完成的 sensing point ID，供外部记录奖励
        """
        completed_id = self.sensing_point_id
        self.sensing_target = None
        self.sensing_point_id = None
        return completed_id
    
    def cancel_sensing_task(self) -> Optional[int]:
        """
        取消 sensing 任务（超时或其他原因）
        
        Returns:
            被取消的 sensing point ID
        """
        return self.complete_sensing_task()
    
    def has_sensing_task(self) -> bool:
        """检查是否有 sensing 任务"""
        return self.sensing_target is not None
    
    def is_available_for_sensing(self) -> bool:
        """
        判断是否可接受 sensing 任务
        
        不可接受的情况：
        - 正在充电
        - 电量不足强制返回中
        - 已有 sensing 任务
        """
        if self.task_status in [UAVTaskStatus.CHARGING, UAVTaskStatus.LOW_BATTERY_RETURN]:
            return False
        if self.has_sensing_task():
            return False
        return True
    
    def get_effective_target(self) -> Optional[Tuple[float, float]]:
        """
        获取当前有效目标位置（考虑 sensing 优先级）
        
        优先级：
        1. sensing_target（如果有）
        2. target_location（订单相关）
        
        注意：LOW_BATTERY 和 CHARGING 的处理在 uav_environment.step() 中
        
        Returns:
            目标位置 (lat, lng) 或 None
        """
        # 优先级 1：sensing 任务
        if self.sensing_target is not None:
            return self.sensing_target
        
        # 优先级 2：订单任务
        return self.target_location

    def assign_order(self, order: Order, assignment_time: datetime):
        """分配订单给UAV"""
        self.current_order = order
        self.task_status = UAVTaskStatus.ASSIGNED
        self.target_location = order.start_location
        self.last_status_change = assignment_time

        # 更新订单状态
        order.assign_to_uav(self.uav_id, assignment_time)

    def start_delivery(self, pickup_time: datetime, min_window: int = 30, max_window: int = 50):
        """
        开始配送任务（UAV到达起点并取到订单）
        
        Args:
            pickup_time: UAV到达起点并取到订单的时间
            min_window: 最小时间窗长度（分钟）
            max_window: 最大时间窗长度（分钟）
        """
        if self.task_status == UAVTaskStatus.ASSIGNED:
            self.task_status = UAVTaskStatus.DELIVERY
            self.target_location = self.current_order.end_location
            self.last_status_change = pickup_time
            
            # 设置订单的start_time并计算时间窗（从pickup_time开始）
            self.current_order.start_delivery(pickup_time, min_window, max_window)

    def complete_delivery(self, completion_time: datetime):
        """完成配送任务"""
        if self.task_status == UAVTaskStatus.DELIVERY:
            self.task_status = UAVTaskStatus.RETURNING
            self.target_location = self.depot_location
            self.last_status_change = completion_time
            self.current_order.complete_delivery(completion_time)

    def return_to_depot(self, return_time: datetime):
        """开始返回仓库"""
        if self.task_status in [UAVTaskStatus.DELIVERY, UAVTaskStatus.IDLE]:
            self.task_status = UAVTaskStatus.RETURNING
            self.target_location = self.depot_location
            self.last_status_change = return_time

    def arrive_at_depot(self, arrival_time: datetime):
        """到达仓库"""
        if self.task_status == UAVTaskStatus.RETURNING:
            self.task_status = UAVTaskStatus.IDLE
            self.target_location = None
            self.current_order = None
            self.last_status_change = arrival_time

    def force_return_due_to_low_battery(self, return_time: datetime):
        """因电量不足强制返回"""
        self.task_status = UAVTaskStatus.LOW_BATTERY_RETURN
        self.target_location = self.depot_location
        self.last_status_change = return_time

    def start_charging(self, start_time: datetime, charging_duration_minutes: int):
        """
        开始充电
        
        Args:
            start_time: 开始充电的时间
            charging_duration_minutes: 充电持续时间（分钟）
        """
        if self.task_status in [UAVTaskStatus.RETURNING, UAVTaskStatus.LOW_BATTERY_RETURN]:
            self.task_status = UAVTaskStatus.CHARGING
            self.target_location = None  # 充电时没有目标位置
            self.last_status_change = start_time
            self.charging_start_time = start_time
            self.charging_duration_minutes = charging_duration_minutes
            # 注意：current_order保持不变，充电完成后可能需要继续执行

    def finish_charging(self, finish_time: datetime):
        """
        完成充电
        
        Returns:
            bool: 是否成功完成充电
        """
        if self.task_status == UAVTaskStatus.CHARGING:
            self.task_status = UAVTaskStatus.IDLE
            self.target_location = None
            self.last_status_change = finish_time
            # 清空充电相关属性
            self.charging_start_time = None
            self.charging_duration_minutes = None
            # 注意：如果之前有订单，可能需要重新分配或取消
            # 这里先清空订单，让系统重新分配
            self.current_order = None
            return True
        return False

    def is_charging(self) -> bool:
        """检查是否正在充电"""
        return self.task_status == UAVTaskStatus.CHARGING

    def get_charging_progress(self, current_time: datetime) -> float:
        """
        获取充电进度（0.0到1.0）
        
        Args:
            current_time: 当前时间
            
        Returns:
            float: 充电进度（0.0到1.0），如果不在充电则返回None
        """
        if not self.is_charging() or not hasattr(self, 'charging_start_time') or self.charging_start_time is None:
            return None
        
        elapsed_minutes = (current_time - self.charging_start_time).total_seconds() / 60.0
        progress = min(1.0, elapsed_minutes / self.charging_duration_minutes)
        return progress

    def is_available_for_assignment(self) -> bool:
        """检查UAV是否可分配新订单"""
        return self.task_status == UAVTaskStatus.IDLE

    def get_current_target(self) -> Optional[Tuple[float, float]]:
        """获取当前目标位置"""
        return self.target_location

    def __str__(self):
        return (f"UAV {self.uav_id}: {self.task_status.value}, "
                f"Order: {self.current_order.order_id if self.current_order else 'None'}")


def generate_order_time_window(pickup_time: datetime,
                              min_window: int = 30,
                              max_window: int = 50) -> Tuple[datetime, datetime]:
    """
    生成订单时间窗（从UAV取到订单的时间开始计算）

    Args:
        pickup_time: UAV取到订单的时间（时间窗起点）
        min_window: 最小时间窗长度（分钟）
        max_window: 最大时间窗长度（分钟）

    Returns:
        (start_time, end_time) 时间窗，start_time = pickup_time
    """
    # 随机生成时间窗长度
    window_length = random.randint(min_window, max_window)

    # 时间窗开始时间：从pickup_time开始（UAV取到订单的时间）
    start_time = pickup_time

    # 时间窗结束时间：start_time + window_length
    end_time = start_time + timedelta(minutes=window_length)

    return start_time, end_time

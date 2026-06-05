import gymnasium as gym
import numpy as np
from gymnasium import spaces
from config import UAV_ENV_CONFIG, REWARD_CONFIG, WEATHER_CONFIG, TASK_CONFIG, DATASET_CONFIG, SENSING_CONFIG
import pandas as pd
from datetime import datetime, timedelta
import random
import torch
from typing import Dict, List, Optional, Tuple
from .order_system import Order, UAVTaskManager, OrderStatus, UAVTaskStatus, generate_order_time_window
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.sensing_reward import SensingRewardInterface
from utilities.geo_utils import (
    euclidean_distance_meters,
    delta_meters_to_lat_lng,
    lat_lng_delta_to_meters,
    meters_to_lat_degrees,
    meters_to_lng_degrees
)
from datasets.dataset_manager import DatasetManager, EpisodeData


class UAVEnvironment(gym.Env):
    """
    UAV环境，基于用户定义的问题架构

    State: s_t = (V_t, Wind_t, Precipitation_t)
    - V_t = (lat_v, lng_v, e_v): UAV 的位置和剩余电量
    - Wind_t: 风指标
    - Precipitation_t: 降水指标

    Action: a_t = <speed>
    - speed: 速度标量（0~1），乘以max_speed得到实际速度（m/min）
    - 飞行方向由环境根据当前任务状态自动决定（朝向目标点）
    - 是否返回仓库由环境内部逻辑决定：
      当电量低于阈值时强制返回仓库
    """

    def __init__(self,
                 config=None,
                 region_bounds=None,
                 max_steps=None,
                 low_battery_threshold=None,
                 depot_location=None,
                 battery_consumption_rate=None,
                 max_movement=None,
                 num_uavs=None,
                 dataset_path=None,
                 enable_orders=None,
                 order_time_window_min=None,
                 order_time_window_max=None,
                 # 新增：时间步长和按天episode相关配置
                 step_minutes: Optional[int] = None,
                 dataset_manager: Optional[DatasetManager] = None,
                 episode_sampling_mode: str = "random",
                 region_id: Optional[int] = None,
                 num_orders_per_episode: Optional[int] = None):
        super(UAVEnvironment, self).__init__()

        # 如果没有提供 config，使用默认配置
        if config is None:
            config = UAV_ENV_CONFIG.copy()

        # 先设置配置参数
        self.reward_config = REWARD_CONFIG.copy()
        self.weather_config = WEATHER_CONFIG.copy()
        self.task_config = TASK_CONFIG.copy()
        self.dataset_config = DATASET_CONFIG.copy()
        self.sensing_config = SENSING_CONFIG.copy()

        # 更新配置为传入的值
        for key, value in config.items():
            if key in self.reward_config:
                self.reward_config[key] = value
            if key in self.weather_config:
                self.weather_config[key] = value
            if key in self.task_config:
                self.task_config[key] = value
            if key in self.dataset_config:
                self.dataset_config[key] = value
            if key in self.sensing_config:
                self.sensing_config[key] = value

        # 从 config 或参数中读取基本参数（用于加载数据集）
        self.enable_orders = enable_orders if enable_orders is not None else config.get("enable_orders", True)
        self.order_time_window_min = order_time_window_min if order_time_window_min is not None else config.get("order_time_window_min", 30)
        self.order_time_window_max = order_time_window_max if order_time_window_max is not None else config.get("order_time_window_max", 50)

        # Weather lookup cache (must exist before _load_real_datasets builds it)
        self._weather_cache_ready = False
        self._weather_minutes: Optional[np.ndarray] = None
        self._weather_precip: Optional[np.ndarray] = None
        self._weather_wind_speed: Optional[np.ndarray] = None
        self._weather_wind_dir: Optional[np.ndarray] = None

        # 加载真实数据集
        dataset_path = dataset_path or self.dataset_config.get("dataset_path")
        self._load_real_datasets(dataset_path)

        # 从 config 或参数中读取环境参数
        if region_bounds is not None:
            self.region_bounds = region_bounds
        elif config.get("region_bounds") is not None:
            self.region_bounds = tuple(config["region_bounds"])
        else:
            self.region_bounds = self._calculate_region_bounds()
        self.max_steps = max_steps if max_steps is not None else config.get("max_steps", 1000)
        self.low_battery_threshold = low_battery_threshold if low_battery_threshold is not None else config.get("low_battery_threshold", 0.3)
        self.depot_location = depot_location if depot_location is not None else config.get("depot_location", (0.0, 0.0))
        self.battery_consumption_rate = battery_consumption_rate if battery_consumption_rate is not None else config.get("battery_consumption_rate", 0.01)
        self.max_movement = max_movement if max_movement is not None else config.get("max_movement", 10.0)
        self.num_uavs = num_uavs if num_uavs is not None else config.get("num_uavs", 1)

        # 时间步长（分钟）：用于控制每个step推进的仿真时间
        self.step_minutes = step_minutes if step_minutes is not None else config.get("step_minutes", 5)

        # 速度控制相关配置
        self.max_speed = config.get("max_speed", 500.0)  # 最大速度 (m/min)
        self.min_speed = config.get("min_speed", 0.0)    # 最小速度 (m/min)
        self.slow_down_distance = config.get("slow_down_distance", 300.0)  # 减速距离阈值 (m)
        self.slow_down_speed = config.get("slow_down_speed", 20.0)  # 减速后的速度 (m/min)
        
        # 电池容量（1000 kJ = 1e6 J），与 energy_penalty 共用同一套能量公式
        self.battery_capacity = config.get("battery_capacity", 1_000_000.0)

        # 充电相关配置
        self.charging_time_minutes = config.get("charging_time_minutes", 10)
        self.auto_charge_threshold = config.get("auto_charge_threshold", 0.3)

        # Daily episode 相关配置
        self.episode_sampling_mode = episode_sampling_mode
        self.region_id = region_id if region_id is not None else self.dataset_config.get("region_id", 0)
        self.num_orders_per_episode = num_orders_per_episode
        self.dataset_manager = dataset_manager
        self.allowed_dates: Optional[List[str]] = config.get("allowed_dates", None) if config else None
        self.preloaded_episodes: Optional[Dict[str, EpisodeData]] = None
        self.current_episode: Optional[EpisodeData] = None
        self.current_episode_date: Optional[str] = None

        # 订单奖励追踪：用于检测订单完成和超时事件
        self.previous_completed_count = 0
        self.previous_timed_out_count = 0

        # Shaping reward 追踪：记录每个UAV上一步到目标点的距离和目标位置
        # 用于计算距离变化：r_shape = k * (d_t - d_{t+1})
        self.previous_target_distances = {}  # {uav_id: distance}
        self.previous_targets = {}  # {uav_id: target_position or None}

        # 调试打印配置
        self.debug_print_actions = config.get("debug_print_actions", False)
        self.print_action_freq = config.get("print_action_freq", 10)  # 每N步打印一次

        # 从其他配置中读取相关参数（其余配置已在前面设置）

        # 订单系统配置
        self.enable_orders = enable_orders if enable_orders is not None else config.get("enable_orders", True)
        self.order_time_window_min = order_time_window_min if order_time_window_min is not None else config.get("order_time_window_min", 30)
        self.order_time_window_max = order_time_window_max if order_time_window_max is not None else config.get("order_time_window_max", 50)
        self.order_completion_distance = self.task_config.get("order_completion_distance", 10)
        self.max_pending_orders = self.task_config.get("max_pending_orders", 50)
        
        # Sensing task 完成距离（米）：UAV 到达 sensing 点的判定距离
        self.sensing_completion_distance = self.sensing_config.get("sensing_completion_distance", 100.0)
        
        # 吸附距离（米）：当UAV距离目标小于此值时，自动吸附到目标点（已废弃，用精度奖励替代）
        self.snap_distance = config.get("snap_distance", 500.0)

        # 观察空间：每个UAV的状态 (lat, lng, battery, wind_speed, wind_direction, precipitation,
        # target_lat, target_lng, task_status)
        # 总维度 = 9 * num_uavs
        single_uav_obs_low = np.array([
            self.region_bounds[0],  # min_lat (基于真实GPS数据)
            self.region_bounds[2],  # min_lng (基于真实GPS数据)
            0.0,                   # battery min
            0.0,                   # wind_speed min (基于天气数据)
            0.0,                   # wind_direction min (基于天气数据)
            0.0,                   # precipitation min (基于天气数据)
            self.region_bounds[0],  # target_lat min
            self.region_bounds[2],  # target_lng min
            0.0                    # task_status min (0=idle, 1=assigned, 2=delivery, 3=returning, 4=charging, 5=low_battery_return)
        ])
        
        single_uav_obs_high = np.array([
            self.region_bounds[1],  # max_lat (基于订单GPS数据)
            self.region_bounds[3],  # max_lng (基于订单GPS数据)
            self.battery_capacity,  # battery max (J)
            15.0,                  # wind_speed max (基于天气数据≈11.3，留余量)
            360.0,                 # wind_direction max (基于天气数据)
            20.0,                  # precipitation max (基于天气数据≈14.7，留余量)
            self.region_bounds[1],  # target_lat max
            self.region_bounds[3],  # target_lng max
            5.0                    # task_status max (0=idle, 1=assigned, 2=delivery, 3=returning, 4=charging, 5=low_battery_return)
        ])

        # 为每个UAV重复状态边界
        obs_low = np.tile(single_uav_obs_low, self.num_uavs)
        obs_high = np.tile(single_uav_obs_high, self.num_uavs)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # 动作空间：所有UAV的速度控制
        # 每个UAV只需要1个动作维度（速度比例），总维度 = num_uavs
        # agent 输出范围 [0, 1]，乘以 max_speed 得到实际速度（m/min）
        # 例如：agent 输出 0.5 → 0.5 * max_speed = 0.5 * 500 = 250 m/min
        # 飞行方向由环境根据当前任务状态自动决定
        action_low = np.full(self.num_uavs, 0.0)
        action_high = np.full(self.num_uavs, 1.0)

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32,
        )

        # 环境状态
        self.current_step = 0
        self.state = None
        self.current_time = None  # 当前仿真时间
        self.previous_reward = 0.0  # 上一个step的reward，用于计算增量reward

        # UAV状态管理
        self.uav_states = []  # 每个UAV的状态列表
        self.uav_positions = []  # 每个UAV的位置 [(lat, lng), ...]
        self.uav_batteries = []  # 每个UAV的电量 [battery, ...]

        # 订单系统管理
        if self.enable_orders:
            self.uav_task_managers = []  # 每个UAV的任务管理器
            self.pending_orders = []     # 待处理的订单队列
            self.active_orders = []      # 正在执行的订单
            self.completed_orders = []   # 已完成的订单
            self.timed_out_orders = []   # 超时的订单

            # 订单统计
            self.total_orders_created = 0
            self.total_orders_completed = 0
            self.total_orders_timed_out = 0

        # 初始化感知覆盖系统
        self.sensing_reward_system = SensingRewardInterface(
            window_hours=self.sensing_config.get("window_hours", 2),
            grid_rows=self.sensing_config.get("grid_rows", 20),
            grid_cols=self.sensing_config.get("grid_cols", 20),
            region_bounds=self.region_bounds,
            alpha=self.sensing_config.get("alpha", 0.7)
        )

        # 更新区域边界到感知系统（确保使用最新边界）
        self.sensing_reward_system.update_region_bounds(self.region_bounds)

    def _load_real_datasets(self, dataset_path):
        """加载真实数据集"""
        if dataset_path is None:
            # 从config中获取数据集路径
            dataset_path = self.dataset_config.get("dataset_path", "datasets")

        try:
            # 获取配置参数
            orders_filename = self.dataset_config.get("orders_filename", "hangzhou_region0_101_MayToJuly.csv")
            weather_filename = self.dataset_config.get("weather_filename", "hourly_data_region101_hz_May_June_July_2022.csv")
            region_id = self.dataset_config.get("region_id", 0)
            use_real_datasets = self.dataset_config.get("use_real_datasets", True)

            if not use_real_datasets:
                print("Real datasets disabled, using synthetic data")
                self.orders_df = None
                self.weather_df = None
                self._clear_weather_cache()
                return

            # 加载订单数据
            orders_path = f"{dataset_path}/{orders_filename}"
            self.orders_df = pd.read_csv(orders_path)
            region_orders = self.orders_df[self.orders_df['region_id'] == region_id]
            print(f"Loaded {len(region_orders)} orders for region {region_id}")

            # 加载天气数据
            weather_path = f"{dataset_path}/{weather_filename}"
            self.weather_df = pd.read_csv(weather_path)
            print(f"Loaded {len(self.weather_df)} weather records")
            self._build_weather_cache()

            # 如果启用订单系统，预处理订单数据
            if self.enable_orders and self.orders_df is not None:
                self._preprocess_orders()

        except FileNotFoundError as e:
            print(f"Warning: Dataset files not found: {e}")
            print("Falling back to synthetic data generation")
            self.orders_df = None
            self.weather_df = None
            self._clear_weather_cache()

    def _preprocess_orders(self):
        """预处理订单数据，生成时间窗等"""
        if self.orders_df is None:
            return

        region_id = self.dataset_config.get("region_id", 0)
        region_orders = self.orders_df[self.orders_df['region_id'] == region_id]

        # 为每个订单生成时间窗
        self.processed_orders = []
        for _, order_row in region_orders.iterrows():
            # 解析订单接受时间
            accept_time = pd.to_datetime(order_row['accept_time'])
            if accept_time.tz is not None:
                accept_time = accept_time.tz_convert('UTC').tz_localize(None)
            else:
                # 调整年份到2022年以匹配天气数据
                accept_time = accept_time.replace(year=2022)

            # 生成时间窗（占位符，会在UAV取到订单时基于pickup_time重新计算）
            # 这里使用accept_time作为占位符
            time_window = generate_order_time_window(
                pickup_time=accept_time,  # 占位符，实际时间窗会在start_delivery时重新计算
                min_window=self.order_time_window_min,
                max_window=self.order_time_window_max
            )

            # 创建订单对象
            order = Order(
                order_id=str(order_row['order_id']),
                start_location=(order_row['region_center_lat'], order_row['region_center_lng']),
                end_location=(order_row['delivery_gps_lat'], order_row['delivery_gps_lng']),
                time_window=time_window,
                accept_time=accept_time
            )

            self.processed_orders.append(order)

        print(f"Preprocessed {len(self.processed_orders)} orders with time windows")

    def _calculate_region_bounds(self):
        """基于真实订单数据计算区域边界"""
        if self.orders_df is None:
            # 回退到默认边界
            return (-100, 100, -100, 100)

        # 从config获取region_id
        region_id = self.dataset_config.get("region_id", 0)

        # 筛选指定region的订单
        region_orders = self.orders_df[self.orders_df['region_id'] == region_id]

        if len(region_orders) == 0:
            region_id = self.dataset_config.get("region_id", 0)
            print(f"Warning: No orders found for region {region_id}, using default bounds")
            return (-100, 100, -100, 100)

        # 计算边界
        min_lat = region_orders['delivery_gps_lat'].min()
        max_lat = region_orders['delivery_gps_lat'].max()
        min_lng = region_orders['delivery_gps_lng'].min()
        max_lng = region_orders['delivery_gps_lng'].max()

        # 计算正方形边界
        lat_center = (min_lat + max_lat) / 2
        lng_center = (min_lng + max_lng) / 2
        lat_range = max_lat - min_lat
        lng_range = max_lng - min_lng
        max_range = max(lat_range, lng_range)

        # 扩展为正方形并添加缓冲区
        buffer = max_range * 0.1
        square_min_lat = lat_center - max_range / 2 - buffer
        square_max_lat = lat_center + max_range / 2 + buffer
        square_min_lng = lng_center - max_range / 2 - buffer
        square_max_lng = lng_center + max_range / 2 + buffer

        bounds = (square_min_lat, square_max_lat, square_min_lng, square_max_lng)
        print(f"Calculated region bounds: {bounds}")
        return bounds

    def _clear_weather_cache(self) -> None:
        self._weather_cache_ready = False
        self._weather_minutes = None
        self._weather_precip = None
        self._weather_wind_speed = None
        self._weather_wind_dir = None

    def _build_weather_cache(self) -> None:
        """Pre-index hourly weather CSV for O(log n) nearest-neighbor lookup."""
        self._clear_weather_cache()
        if self.weather_df is None or len(self.weather_df) == 0:
            return

        df = self.weather_df
        if 'datetime' in df.columns:
            datetimes = pd.to_datetime(df['datetime'])
        else:
            datetimes = pd.to_datetime(df['date'])
        if datetimes.dt.tz is not None:
            datetimes = datetimes.dt.tz_convert(None)

        epoch = pd.Timestamp('1970-01-01')
        minutes = ((datetimes - epoch) // pd.Timedelta('1min')).astype(np.int64).to_numpy()
        order = np.argsort(minutes, kind='mergesort')
        self._weather_minutes = minutes[order]
        self._weather_precip = df['precipitation'].to_numpy(dtype=np.float64)[order]
        self._weather_wind_speed = df['wind_speed_100m'].to_numpy(dtype=np.float64)[order]
        self._weather_wind_dir = df['wind_direction_100m'].to_numpy(dtype=np.float64)[order]
        self._weather_cache_ready = True

    def _weather_index_for_timestamp(self, timestamp) -> int:
        query_min = int(
            (pd.Timestamp(timestamp) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1min')
        )
        idx = int(np.searchsorted(self._weather_minutes, query_min))
        if idx <= 0:
            return 0
        if idx >= len(self._weather_minutes):
            return len(self._weather_minutes) - 1
        if abs(int(self._weather_minutes[idx]) - query_min) < abs(int(self._weather_minutes[idx - 1]) - query_min):
            return idx
        return idx - 1

    def _get_weather_at_time(self, timestamp):
        """根据时间戳获取天气数据"""
        if self.weather_df is None:
            # 回退到随机生成
            return {
                'precipitation': np.random.exponential(0.5),
                'wind_speed': np.random.normal(0, 1),
                'wind_direction': np.random.uniform(0, 360)
            }

        # 确保timestamp也是naive的
        if timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)

        if self._weather_cache_ready:
            best = self._weather_index_for_timestamp(timestamp)
            return {
                'precipitation': float(self._weather_precip[best]),
                'wind_speed': float(self._weather_wind_speed[best]),
                'wind_direction': float(self._weather_wind_dir[best]),
            }

        # Fallback: legacy pandas scan (should not run after successful cache build)
        if 'datetime' not in self.weather_df.columns:
            self.weather_df['datetime'] = pd.to_datetime(self.weather_df['date']).dt.tz_localize(None)
        time_diffs = np.abs((self.weather_df['datetime'] - timestamp).dt.total_seconds() / 60)
        closest_idx = time_diffs.idxmin()
        weather_row = self.weather_df.loc[closest_idx]
        return {
            'precipitation': float(weather_row['precipitation']),
            'wind_speed': float(weather_row['wind_speed_100m']),
            'wind_direction': float(weather_row['wind_direction_100m'])
        }

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)

        # 初始化UAV状态
        self.current_step = 0
        self.previous_reward = 0.0  # 重置上一个step的reward

        # =========================
        # Daily episode 模式：每个episode对应一天的数据
        # =========================
        if self.episode_sampling_mode == "daily" and self.dataset_manager is not None:
            # 优先从内存预加载的 episodes 中随机选取（零 I/O）
            if self.preloaded_episodes and self.allowed_dates is not None:
                pool = [d for d in self.allowed_dates if d in self.preloaded_episodes]
            else:
                pool = []

            if pool:
                selected_date = np.random.choice(pool)
                episode = self.preloaded_episodes[selected_date]
            elif self.allowed_dates is not None:
                available_dates = [d for d in self.allowed_dates 
                                 if d in self.dataset_manager.get_available_dates(str(self.region_id))]
                if not available_dates:
                    raise ValueError(f"No available dates in allowed_dates list for region {self.region_id}")
                if seed is not None:
                    np.random.seed(seed)
                selected_date = np.random.choice(available_dates)
                episode = self.dataset_manager.sample_episode_by_date(
                    date_str=selected_date,
                    region_id=str(self.region_id),
                    num_orders=self.num_orders_per_episode
                )
            else:
                episode = self.dataset_manager.sample_episode(
                    region_id=str(self.region_id),
                    num_orders=self.num_orders_per_episode,
                    mode="daily",
                )
            self.current_episode = episode
            self.current_episode_date = episode.date_str

            # 使用EpisodeData中的region_bounds和depot_location覆盖默认配置
            self.region_bounds = episode.region_bounds.to_tuple()
            self.depot_location = episode.depot_location

            self.sensing_reward_system.update_region_bounds(self.region_bounds)

            # 当天的起始时间作为当前时间
            self.current_time = episode.time_window[0]

            # 基于EpisodeData.orders构建订单系统内部的Order对象列表
            # 注意：这里使用region_center作为起点、delivery_gps作为终点，与greedy baseline保持一致
            self.processed_orders = []
            for ep_order in episode.orders:
                accept_time = ep_order.arrival_time

                # 生成占位时间窗（真正的时间窗会在start_delivery时重新计算）
                time_window = generate_order_time_window(
                    pickup_time=accept_time,
                    min_window=self.order_time_window_min,
                    max_window=self.order_time_window_max,
                )

                internal_order = Order(
                    order_id=str(ep_order.order_id),
                    start_location=(ep_order.region_center_lat, ep_order.region_center_lng),
                    end_location=(ep_order.delivery_lat, ep_order.delivery_lng),
                    time_window=time_window,
                    accept_time=accept_time,
                )
                self.processed_orders.append(internal_order)

            # 按接受时间排序，保证订单按时间逐步激活
            self.processed_orders.sort(key=lambda o: o.accept_time)
        else:
            # =========================
            # 原有随机模式：从整段数据中随机选择一个时间起点
            # =========================
            # 设置初始时间（从订单数据中随机选择一个时间点）
            if self.orders_df is not None and len(self.orders_df) > 0:
                random_order = self.orders_df.sample(random_state=seed).iloc[0]
                # 解析时间并移除时区信息（如果有的话）
                order_time = pd.to_datetime(random_order['accept_time'])
                if order_time.tz is not None:
                    order_time = order_time.tz_convert('UTC').tz_localize(None)
                # 调整年份到2022年以匹配天气数据
                self.current_time = order_time.replace(year=2022)
            else:
                # 回退到默认时间
                self.current_time = datetime(2022, 5, 1, 12, 0, 0)

        # 初始化所有UAV的状态
        self.uav_states = []
        self.uav_positions = []
        self.uav_batteries = []

        # 初始化订单系统
        if self.enable_orders:
            # `uav_task_managers` 可以在多个 episode 之间复用，但订单相关列表和统计
            # 必须在每个 episode 开始时清空，避免跨 episode 累积。
            if not hasattr(self, 'uav_task_managers'):
                self.uav_task_managers = []

            # 每个 episode 重新初始化订单队列和状态列表
            self.pending_orders = []
            self.active_orders = []
            self.completed_orders = []
            self.timed_out_orders = []

            # 重置订单统计
            self.total_orders_created = 0
            self.total_orders_completed = 0
            self.total_orders_timed_out = 0

            # 重置订单奖励追踪
            self.previous_completed_count = 0
            self.previous_timed_out_count = 0

            # 重置 shaping reward 追踪
            self.previous_target_distances.clear()
            self.previous_targets.clear()

            # 如果有预处理的订单数据，添加到待处理队列
            if hasattr(self, 'processed_orders') and self.processed_orders:
                # 按照接受时间排序
                sorted_orders = sorted(self.processed_orders, key=lambda x: x.accept_time)
                self.pending_orders = sorted_orders[:self.max_pending_orders]  # 限制待处理订单数量

            # 确保有足够的任务管理器
            while len(self.uav_task_managers) < self.num_uavs:
                uav_id = len(self.uav_task_managers)
                task_manager = UAVTaskManager(uav_id, self.depot_location)
                self.uav_task_managers.append(task_manager)
            
            # 重置所有UAV任务管理器（清理上一episode的订单状态，避免跨episode保留）
            for task_manager in self.uav_task_managers:
                task_manager.current_order = None
                task_manager.task_status = UAVTaskStatus.IDLE
                task_manager.target_location = None
                task_manager.last_status_change = None
                # 更新仓库位置（可能在新episode中改变）
                task_manager.depot_location = self.depot_location
                # 清理充电相关状态
                if hasattr(task_manager, 'charging_start_time'):
                    task_manager.charging_start_time = None
                if hasattr(task_manager, 'charging_duration_minutes'):
                    task_manager.charging_duration_minutes = None

        # 获取初始天气数据
        weather_data = self._get_weather_at_time(self.current_time)

        for uav_id in range(self.num_uavs):
            # 所有UAV从仓库出发，满电量
            initial_lat, initial_lng = self.depot_location
            initial_battery = self.battery_capacity

            # 为每个UAV创建状态向量
            if self.enable_orders:
                task_manager = self.uav_task_managers[uav_id]
                target_pos = task_manager.get_current_target() or (0.0, 0.0)
                task_status_value = list(UAVTaskStatus).index(task_manager.task_status)
            else:
                target_pos = (0.0, 0.0)
                task_status_value = 0

            uav_state = np.array([
                initial_lat,                    # lat
                initial_lng,                    # lng
                initial_battery,                # battery
                weather_data['wind_speed'],     # wind_speed (共享天气)
                weather_data['wind_direction'], # wind_direction (共享天气)
                weather_data['precipitation'],  # precipitation (共享天气)
                target_pos[0],                  # target_lat
                target_pos[1],                  # target_lng
                task_status_value               # task_status
            ], dtype=np.float32)

            self.uav_states.append(uav_state)
            self.uav_positions.append((initial_lat, initial_lng))
            self.uav_batteries.append(initial_battery)

            # 初始化UAV任务管理器
            if self.enable_orders:
                if len(self.uav_task_managers) <= uav_id:
                    task_manager = UAVTaskManager(uav_id, self.depot_location)
                    self.uav_task_managers.append(task_manager)

        # 重置感知覆盖系统
        self.sensing_reward_system.reset()

        # 合并所有UAV的状态为单一的状态向量
        self.state = np.concatenate(self.uav_states)

        return self.state, {}

    def step(self, action):
        """执行一步动作"""
        self.current_step += 1

        # 记录订单完成数量（用于检测新完成的订单）
        orders_completed_before = self.total_orders_completed if self.enable_orders else 0

        # 订单系统：处理订单分派和状态更新
        if self.enable_orders:
            self._process_order_system()

        # 解析动作：action 为形状 (num_uavs,) 的向量
        # 每个值对应一个UAV的速度比例 [0, 1]
        # 乘以 max_speed 得到实际速度（m/min）
        uav_speeds = []
        per_uav_speed_ratio = []
        for i in range(self.num_uavs):
            speed_ratio = float(np.clip(action[i], 0.0, 1.0))  # 确保在[0,1]范围内
            target_speed = speed_ratio * self.max_speed  # 得到实际速度 (m/min)
            uav_speeds.append(target_speed)
            per_uav_speed_ratio.append(speed_ratio)

        # 更新时间（使用可配置的step_minutes）
        next_time = self.current_time + timedelta(minutes=self.step_minutes)
        self.current_time = next_time

        # 获取下一时刻的天气数据
        next_weather = self._get_weather_at_time(next_time)

        # 风速矢量（m/min），用于沿航向风分量计算
        _wind_spd_mpm = next_weather['wind_speed'] * 60.0
        # _wind_spd_mpm = next_weather['wind_speed'] 
        _wind_dir_rad = np.radians(next_weather['wind_direction'])
        wind_vec_lat = -_wind_spd_mpm * np.cos(_wind_dir_rad)
        wind_vec_lng = -_wind_spd_mpm * np.sin(_wind_dir_rad)

        # 更新所有UAV的状态
        total_reward = 0
        sensing_reward = 0.0  # 用于单独追踪感知奖励
        shaping_reward_total = 0.0  # 用于单独追踪shaping奖励
        energy_penalty_total = 0.0  # 用于单独追踪能量惩罚
        new_uav_states = []
        new_positions = []
        new_batteries = []
        per_uav_effective_speeds = []
        per_uav_actual_speed = []
        per_uav_battery_consumption = []
        per_uav_energy_penalty = []

        for uav_id, target_speed in enumerate(uav_speeds):
            displacement = 0.0  # 充电/idle/已到位等路径不进入移动分支，需有默认值
            # 获取当前UAV状态
            current_state = self.uav_states[uav_id]
            current_lat, current_lng, current_battery, _, _, _, _, _, _ = current_state

            # 检查是否强制返回仓库（电量低于阈值，阈值为容量比例）
            low_battery_j = self.low_battery_threshold * self.battery_capacity
            forced_return = current_battery < low_battery_j
            step_energy_j = 0.0

            # 检查充电状态（优先于forced_return处理）
            is_charging = False
            if self.enable_orders:
                task_manager = self.uav_task_managers[uav_id]
                is_charging = task_manager.is_charging()

            if is_charging:
                # 充电时UAV不能移动，保持在仓库位置
                next_lat, next_lng = self.depot_location
                actual_speed = 0.0
                effective_speed = 0.0
                
                # 检查充电是否完成
                charging_progress = task_manager.get_charging_progress(self.current_time)
                if charging_progress is not None and charging_progress >= 1.0:
                    # 充电完成
                    task_manager.finish_charging(self.current_time)
                    next_battery = self.battery_capacity  # 恢复满电
                else:
                    # 仍在充电，电量不变
                    next_battery = current_battery
                
                # 充电时不消耗电量，也不移动
                movement_distance = 0.0
            elif forced_return:
                # 强制返回仓库：以最大速度飞回仓库
                target_pos = self.depot_location
                actual_speed = self.max_speed  # 强制返回时使用最大速度
                
                # 计算方向和距离
                distance_to_depot = euclidean_distance_meters(
                    current_lat, current_lng, target_pos[0], target_pos[1]
                )
                
                if distance_to_depot > 0:
                    # 计算方向向量（经纬度）
                    direction_lat = target_pos[0] - current_lat
                    direction_lng = target_pos[1] - current_lng
                    # 转换为米
                    direction_lat_m, direction_lng_m = lat_lng_delta_to_meters(
                        direction_lat, direction_lng, current_lat
                    )
                    direction_magnitude = np.sqrt(direction_lat_m**2 + direction_lng_m**2)
                    
                    # 沿航向风分量（正=顺风）→ 有效速度
                    heading_lat = direction_lat_m / direction_magnitude
                    heading_lng = direction_lng_m / direction_magnitude
                    w_tailwind = wind_vec_lat * heading_lat + wind_vec_lng * heading_lng
                    effective_speed = actual_speed + w_tailwind
                    
                    # 计算位移距离（实际速度 × 时间）
                    displacement = actual_speed * self.step_minutes  # 米
                    displacement = min(displacement, distance_to_depot)  # 不超过目标
                    
                    # 归一化并计算实际位移
                    delta_lat_m = (direction_lat_m / direction_magnitude) * displacement
                    delta_lng_m = (direction_lng_m / direction_magnitude) * displacement
                    
                    # 转换回经纬度位移
                    delta_lat, delta_lng = delta_meters_to_lat_lng(delta_lat_m, delta_lng_m, current_lat)
                    next_lat = current_lat + delta_lat
                    next_lng = current_lng + delta_lng
                else:
                    next_lat, next_lng = target_pos
                    effective_speed = 0.0
                
                movement_distance = euclidean_distance_meters(
                    current_lat, current_lng, next_lat, next_lng
                )

                step_energy_j = self._compute_step_energy_j(actual_speed, effective_speed)

                # 更新任务状态为强制返回
                if self.enable_orders:
                    self.uav_task_managers[uav_id].force_return_due_to_low_battery(self.current_time)
                
                # 更新电量
                next_battery = max(0.0, current_battery - step_energy_j)
            else:
                # 正常移动：方向由任务状态决定，速度由agent控制
                target_pos = self._get_target_for_uav(uav_id)
                effective_speed = 0.0
                
                if target_pos is None:
                    # IDLE状态：没有目标，停在原地
                    next_lat, next_lng = current_lat, current_lng
                    actual_speed = 0.0
                    movement_distance = 0.0
                else:
                    # 有目标：朝目标飞行
                    distance_to_target = euclidean_distance_meters(
                        current_lat, current_lng, target_pos[0], target_pos[1]
                    )
                    
                    # 近目标强制减速
                    if distance_to_target < self.slow_down_distance:
                        actual_speed = min(target_speed, self.slow_down_speed)
                    else:
                        actual_speed = target_speed
                    
                    if distance_to_target > 0 and actual_speed > 0:
                        # 计算方向向量（经纬度）
                        direction_lat = target_pos[0] - current_lat
                        direction_lng = target_pos[1] - current_lng
                        # 转换为米
                        direction_lat_m, direction_lng_m = lat_lng_delta_to_meters(
                            direction_lat, direction_lng, current_lat
                        )
                        direction_magnitude = np.sqrt(direction_lat_m**2 + direction_lng_m**2)
                        
                        # 沿航向风分量（正=顺风）→ 有效速度
                        heading_lat = direction_lat_m / direction_magnitude
                        heading_lng = direction_lng_m / direction_magnitude
                        w_tailwind = wind_vec_lat * heading_lat + wind_vec_lng * heading_lng
                        effective_speed = actual_speed + w_tailwind
                        
                        # 计算位移距离（有效速度 × 时间）
                        displacement = actual_speed * self.step_minutes  # 米
                        displacement = min(displacement, distance_to_target)  # 不超过目标
                        
                        # 归一化并计算实际位移
                        delta_lat_m = (direction_lat_m / direction_magnitude) * displacement
                        delta_lng_m = (direction_lng_m / direction_magnitude) * displacement
                        
                        # 转换回经纬度位移
                        delta_lat, delta_lng = delta_meters_to_lat_lng(delta_lat_m, delta_lng_m, current_lat)
                        next_lat = current_lat + delta_lat
                        next_lng = current_lng + delta_lng
                    else:
                        # 已经到达目标或速度为0
                        next_lat, next_lng = current_lat, current_lng
                        actual_speed = 0.0
                        effective_speed = 0.0

                    # 边界裁剪
                    next_lat = np.clip(next_lat, self.region_bounds[0], self.region_bounds[1])
                    next_lng = np.clip(next_lng, self.region_bounds[2], self.region_bounds[3])

                    # 使用米为单位计算实际移动距离
                    movement_distance = euclidean_distance_meters(
                        current_lat, current_lng, next_lat, next_lng
                    )

                step_energy_j = self._compute_step_energy_j(actual_speed, effective_speed)
                # 更新电量
                next_battery = max(0.0, current_battery - step_energy_j)
            
            battery_consumption = step_energy_j
            energy_penalty = -step_energy_j
            energy_penalty_total += energy_penalty
            total_reward += energy_penalty
            per_uav_effective_speeds.append(float(effective_speed))
            per_uav_actual_speed.append(float(actual_speed))
            per_uav_battery_consumption.append(float(battery_consumption))
            per_uav_energy_penalty.append(float(energy_penalty))

            # 创建新的UAV状态
            if self.enable_orders:
                task_manager = self.uav_task_managers[uav_id]
                target_pos = task_manager.get_current_target() or (next_lat, next_lng)
                task_status_value = list(UAVTaskStatus).index(task_manager.task_status)
            else:
                target_pos = (next_lat, next_lng)
                task_status_value = 0

            new_uav_state = np.array([
                next_lat,
                next_lng,
                next_battery,
                next_weather['wind_speed'],
                next_weather['wind_direction'],
                next_weather['precipitation'],
                target_pos[0],
                target_pos[1],
                task_status_value
            ], dtype=np.float32)

            new_uav_states.append(new_uav_state)
            new_positions.append((next_lat, next_lng))
            new_batteries.append(next_battery)

            # 计算 shaping reward（鼓励UAV朝向目标移动）
            # 在 ASSIGNED 和 DELIVERY 状态时计算，鼓励UAV朝向目标移动
            if self.enable_orders:
                task_manager = self.uav_task_managers[uav_id]
                if task_manager.task_status in [UAVTaskStatus.ASSIGNED, UAVTaskStatus.DELIVERY]:
                    shaping_reward = self._calculate_shaping_reward(
                        uav_id, 
                        self.uav_positions[uav_id],  # 移动前的位置
                        (next_lat, next_lng)  # 移动后的位置
                    )
                    shaping_reward_total += shaping_reward
                    total_reward += shaping_reward

     
                   
        # 打印UAV action和路径信息（用于调试）
        if self.debug_print_actions:
            for uav_id in range(self.num_uavs):
                agent_speed = uav_speeds[uav_id]
                prev_pos = self.uav_positions[uav_id]
                new_pos = new_positions[uav_id]
                prev_battery = self.uav_batteries[uav_id]
                new_battery = new_batteries[uav_id]
                
                # 获取任务状态
                if self.enable_orders:
                    task_status = self.uav_task_managers[uav_id].task_status.value
                    current_order_id = self.uav_task_managers[uav_id].current_order.order_id if self.uav_task_managers[uav_id].current_order else "None"
                else:
                    task_status = "N/A"
                    current_order_id = "N/A"
                
                eff = per_uav_effective_speeds[uav_id]
                wspd = next_weather['wind_speed']
                wdir = next_weather['wind_direction']
                print(f"[Step {self.current_step}] UAV {uav_id}: "
                      f"speed={agent_speed:.2f} m/min, "
                      f"effective_speed={eff:.2f} m/min, "
                      f"wind_speed={wspd:.3f} (dataset), wind_dir={wdir:.1f}°, "
                      f"pos=({prev_pos[0]:.6f}, {prev_pos[1]:.6f}) -> ({new_pos[0]:.6f}, {new_pos[1]:.6f}), "
                      f"battery={prev_battery:.0f}J->{new_battery:.0f}J, "
                      f"status={task_status}, order={current_order_id}")

        # 计算订单相关奖励（在所有UAV处理完后统一计算，因为订单完成/超时是全局事件）
        order_reward, order_completion_revenue, order_timeout_penalty = self._calculate_order_reward()
        total_reward += order_reward

        # 检测是否有新完成的订单（在_process_order_system之后）
        orders_completed_after = self.total_orders_completed if self.enable_orders else 0
        new_completions = orders_completed_after - orders_completed_before

        # 每一步都打印奖励信息（已注释，改由 macro_env 在 macro step 级别统一打印）
        # print(f"[Step {self.current_step}] Orders Reward: {order_reward:.6f}, Sensing Reward: {sensing_reward:.6f}, Shaping Reward: {shaping_reward_total:.6f}, Energy Penalty: {energy_penalty_total:.6f}, Total Reward: {total_reward:.6f}")
        
        # # 如果有新完成的订单，打印提示
        # if new_completions > 0:
        #     print(f"[Step {self.current_step}] 订单已经送达！完成订单数量: {new_completions}")

        # 更新UAV状态
        self.uav_states = new_uav_states
        self.uav_positions = new_positions
        self.uav_batteries = new_batteries

        # ========== Sensing Task 完成检测 ==========
        # 检测 UAV 是否到达 sensing 点，如果是则完成 sensing 任务并计算奖励
        sensing_completed_ids = []
        if self.enable_orders:
            for uav_id in range(self.num_uavs):
                task_manager = self.uav_task_managers[uav_id]
                if task_manager.has_sensing_task():
                    uav_pos = new_positions[uav_id]
                    sens_pos = task_manager.sensing_target
                    dist = euclidean_distance_meters(
                        uav_pos[0], uav_pos[1], sens_pos[0], sens_pos[1]
                    )
                    
                    if dist <= self.sensing_completion_distance:
                        # 到达 sensing 点，完成任务
                        completed_id = task_manager.complete_sensing_task()
                        if completed_id is not None:
                            sensing_completed_ids.append(completed_id)
                            
                            # 记录收集前的 entropy 值
                            old_entropy_reward = self.sensing_reward_system.calculate_current_reward()
                            
                            # 收集传感数据（使用现有的 sensing_reward_system）
                            self.sensing_reward_system.collect_sensor_data(
                                uav_pos[0], uav_pos[1], self.current_time
                            )
                            
                            # 计算收集后的 entropy 值和增量奖励
                            new_entropy_reward = self.sensing_reward_system.calculate_current_reward()
                            incremental_sensing_reward = new_entropy_reward - old_entropy_reward
                            
                            # 应用下层 sensing 熵增量放大系数（sensing_entropy_weight，与上层 macro 的 sensing_weight 语义不同）
                            sensing_weight = self.reward_config.get("sensing_entropy_weight", 2000.0)
                            weighted_sensing_reward = incremental_sensing_reward * sensing_weight
                            sensing_reward += weighted_sensing_reward
                            total_reward += weighted_sensing_reward
                            
                            # print(f"[Step {self.current_step}] UAV {uav_id} 完成 sensing 任务！"
                            #       f" sensing_point_id={completed_id}, 位置=({sens_pos[0]:.6f}, {sens_pos[1]:.6f}),"
                            #       f" sensing_reward_increment={weighted_sensing_reward:.4f}")

        # 合并所有UAV的状态为单一的状态向量
        self.state = np.concatenate(self.uav_states)

        # 检查终止条件
        terminated = False
        truncated = self.current_step >= self.max_steps

        # 如果到达episode时间窗口的结束时间（24:00），也结束episode
        if self.current_episode is not None and self.current_time >= self.current_episode.time_window[1]:
            truncated = True

        # 如果任何UAV电量为0，结束episode
        if any(battery <= 0 for battery in self.uav_batteries):
            terminated = True

        # TODO: 可以添加其他终止条件，比如所有订单完成等

        info = {
            "uav_positions": self.uav_positions.copy(),
            "uav_batteries": self.uav_batteries.copy(),
            "forced_returns": [
                battery < self.low_battery_threshold * self.battery_capacity
                for battery in self.uav_batteries
            ],
            "current_time": self.current_time,
            "sensing_completed_ids": sensing_completed_ids,        # Preference Stage 使用
            "step_sensing_entropy_reward": sensing_reward,              # 熵函数计算的 sensing 奖励（已×sensing_weight，供 macro_env 统计）
            "step_delivery_reward": order_reward,                      # 订单奖励：完成收益+超时惩罚（含 order_weight）
            "step_delivery_completion_reward": order_completion_revenue,  # 纯配送完成收益（>= 0，含 order_weight），业务 objective 用
            "step_order_timeout_penalty": order_timeout_penalty,       # 纯订单超时惩罚（<= 0，含 order_weight），辅助指标
            "step_energy_penalty": energy_penalty_total,               # 能耗惩罚（负数，供 macro_env 统计）
            "step_shaping_reward": shaping_reward_total,               # shaping 辅助奖励（供 macro_env 统计）
            # ---- 每个 UAV 的逐步物理量（case study 用） ----
            # speed_ratio: 动作输入 [0,1]
            # actual_speed: 实际速度 (m/min)
            # effective_speed: 沿航向风修正后的有效速度 (m/min)
            # battery_consumption: 本步电量消耗增量（J）
            # uav_energy_penalty_per_uav: 本步该 UAV 的能耗惩罚（负数，奖励空间）
            "uav_speed_ratio": per_uav_speed_ratio,
            "uav_actual_speed": per_uav_actual_speed,
            "uav_effective_speed": per_uav_effective_speeds,
            "uav_battery_consumption": per_uav_battery_consumption,
            "uav_energy_penalty_per_uav": per_uav_energy_penalty,
        }

        # 如果是daily episode模式，附带当前episode日期信息，方便上层记录
        if self.current_episode_date is not None:
            info["date_str"] = self.current_episode_date

        # 添加订单系统信息
        if self.enable_orders:
            # 计算总订单数（进入episode的订单总数）
            total_orders_in_episode = 0
            if hasattr(self, 'processed_orders') and self.processed_orders:
                total_orders_in_episode = len(self.processed_orders)
            elif self.current_episode is not None and hasattr(self.current_episode, 'orders'):
                total_orders_in_episode = len(self.current_episode.orders)
            
            # 基于当前 episode 中订单的真实状态统计完成/超时数量，避免计数与总订单数不一致
            if hasattr(self, 'processed_orders') and self.processed_orders:
                completed_in_episode = sum(
                    1 for o in self.processed_orders
                    if getattr(o, "status", None) == OrderStatus.COMPLETED
                )
                timed_out_in_episode = sum(
                    1 for o in self.processed_orders
                    if getattr(o, "status", None) == OrderStatus.TIMEOUT
                )
            else:
                # 回退到全局统计计数（理论上应与订单状态一致）
                completed_in_episode = self.total_orders_completed
                timed_out_in_episode = self.total_orders_timed_out

            info.update({
                "pending_orders": len(self.pending_orders),
                "active_orders": len(self.active_orders),
                "completed_orders": completed_in_episode,
                "timed_out_orders": timed_out_in_episode,
                "total_orders_completed": completed_in_episode,
                "total_orders_timed_out": timed_out_in_episode,
                "total_orders_in_episode": total_orders_in_episode,  # 新增：进入episode的订单总数
                "uav_task_status": [str(task_manager.task_status.value) for task_manager in self.uav_task_managers]
            })
            
            # Episode结束时打印订单统计信息（基于本 episode 内订单状态）
            if terminated or truncated:
                completion_rate = (
                    completed_in_episode / total_orders_in_episode * 100
                ) if total_orders_in_episode > 0 else 0.0
                date_info = f", date={self.current_episode_date}" if self.current_episode_date else ""
                print(f"[Episode End{date_info}] Step={self.current_step}, "
                      f"总订单数={total_orders_in_episode}, "
                      f"已完成订单数={completed_in_episode}, "
                      f"超时订单数={timed_out_in_episode}, "
                      f"完成率={completion_rate:.2f}%")

        return self.state, total_reward, terminated, truncated, info

    def _compute_step_energy_j(self, actual_speed: float, effective_speed: float) -> float:
        """计算单步能量消耗（J），与 energy_penalty 共用同一公式。"""
        coeff = self.reward_config.get("energy_penalty_coeff", 0.0001)
        return (
            coeff * (actual_speed ** 2) * self.step_minutes
            + coeff * 0.0001 * (effective_speed ** 3) * self.step_minutes
        )

    def _get_target_for_uav(self, uav_id: int) -> Optional[Tuple[float, float]]:
        """
        根据UAV的当前任务状态返回目标位置
        
        目标优先级：
        1. sensing_target（Preference Stage 分配的感知任务）
        2. order_target（订单相关任务）
        
        注意：LOW_BATTERY_RETURN 和 CHARGING 状态在 step() 中有更高优先级的处理
        
        Args:
            uav_id: UAV ID
            
        Returns:
            目标位置 (lat, lng)，如果没有目标则返回 None
        """
        if not self.enable_orders:
            return None
            
        task_manager = self.uav_task_managers[uav_id]
        
        # ========== 优先级 1：sensing 任务 ==========
        # 如果有 sensing 任务，优先飞向 sensing 点
        if task_manager.has_sensing_task():
            return task_manager.sensing_target
        
        # ========== 优先级 2：订单任务（原有逻辑） ==========
        if task_manager.task_status == UAVTaskStatus.IDLE:
            # 空闲状态：没有目标
            return None
        elif task_manager.task_status == UAVTaskStatus.ASSIGNED:
            # 已分配订单：目标是订单起点（取货点）
            if task_manager.current_order is not None:
                return task_manager.current_order.start_location
            return None
        elif task_manager.task_status == UAVTaskStatus.DELIVERY:
            # 配送中：目标是订单终点（送达点）
            if task_manager.current_order is not None:
                return task_manager.current_order.end_location
            return None
        elif task_manager.task_status in [UAVTaskStatus.RETURNING, UAVTaskStatus.LOW_BATTERY_RETURN]:
            # 返回仓库
            return self.depot_location
        elif task_manager.task_status == UAVTaskStatus.CHARGING:
            # 充电中：没有移动目标
            return None
        else:
            return None

    def _process_order_system(self):
        """处理订单系统逻辑"""
        # 检查是否有新订单到达
        self._check_new_orders()

        # 尝试分配订单给空闲的UAV
        self._assign_orders_to_uavs()

        # 检查订单完成情况
        self._check_order_completion()

        # 检查超时订单
        self._check_timeout_orders()

    def _check_new_orders(self):
        """检查是否有新订单到达"""
        # 从待处理订单中找到当前时间点应该激活的订单
        new_active_orders = []
        remaining_pending = []

        for order in self.pending_orders:
            if order.accept_time <= self.current_time:
                new_active_orders.append(order)
            else:
                remaining_pending.append(order)

        self.pending_orders = remaining_pending

        # 将新激活的订单添加到活跃订单列表
        for order in new_active_orders:
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.PENDING  # 保持等待状态，等待UAV分配
                self.active_orders.append(order)

    def _assign_orders_to_uavs(self):
        """尝试分配订单给空闲的UAV"""
        # 找到空闲的UAV
        available_uavs = []
        for uav_id, task_manager in enumerate(self.uav_task_managers):
            if task_manager.is_available_for_assignment():
                available_uavs.append(uav_id)

        # 为每个空闲UAV分配订单
        for uav_id in available_uavs:
            if self.active_orders:
                # 找到最早的等待订单
                pending_orders = [order for order in self.active_orders
                                if order.status == OrderStatus.PENDING]
                if pending_orders:
                    # 按接受时间排序，选择最早的订单
                    pending_orders.sort(key=lambda x: x.accept_time)
                    order_to_assign = pending_orders[0]

                    # 分配订单给UAV
                    self.uav_task_managers[uav_id].assign_order(order_to_assign, self.current_time)
                    order_to_assign.status = OrderStatus.ASSIGNED

    def _check_order_completion(self):
        """检查订单完成情况"""
        for uav_id, task_manager in enumerate(self.uav_task_managers):
            if task_manager.current_order is not None:
                order = task_manager.current_order
                uav_pos = self.uav_positions[uav_id]

                # 检查是否到达目标位置
                if task_manager.task_status == UAVTaskStatus.ASSIGNED:
                    # ASSIGNED状态：检查是否到达订单起点
                    target_pos = order.start_location
                    # 使用米为单位计算距离
                    distance = euclidean_distance_meters(
                        uav_pos[0], uav_pos[1], target_pos[0], target_pos[1]
                    )
                    if distance <= self.order_completion_distance:
                        # 到达订单起点，开始配送
                        task_manager.start_delivery(
                            self.current_time,
                            min_window=self.order_time_window_min,
                            max_window=self.order_time_window_max
                        )

                elif task_manager.task_status == UAVTaskStatus.DELIVERY: 
                    # 检查是否到达订单终点
                    target_pos = order.end_location
                    # 使用米为单位计算距离
                    distance = euclidean_distance_meters(
                        uav_pos[0], uav_pos[1], target_pos[0], target_pos[1]
                    )
                    if distance <= self.order_completion_distance:
                        task_manager.complete_delivery(self.current_time)
                        self.completed_orders.append(order)
                        self.total_orders_completed += 1

                elif task_manager.task_status == UAVTaskStatus.RETURNING:
                    # 检查是否返回仓库
                    target_pos = self.depot_location
                    # 使用米为单位计算距离
                    distance = euclidean_distance_meters(
                        uav_pos[0], uav_pos[1], target_pos[0], target_pos[1]
                    )
                    if distance <= self.order_completion_distance:
                        # 检查电量，决定是否需要充电
                        uav_battery = self.uav_batteries[uav_id]
                        if uav_battery < self.auto_charge_threshold * self.battery_capacity:
                            # 电量低，开始充电
                            task_manager.start_charging(
                                self.current_time,
                                self.charging_time_minutes
                            )
                        else:
                            # 电量充足，直接完成返回
                            task_manager.arrive_at_depot(self.current_time)

                elif task_manager.task_status == UAVTaskStatus.LOW_BATTERY_RETURN:
                    # 检查是否返回仓库（低电量强制返回）
                    target_pos = self.depot_location
                    # 使用米为单位计算距离
                    distance = euclidean_distance_meters(
                        uav_pos[0], uav_pos[1], target_pos[0], target_pos[1]
                    )
                    if distance <= self.order_completion_distance:
                        # 低电量返回，必须充电
                        task_manager.start_charging(
                            self.current_time,
                            self.charging_time_minutes
                        )

    def _check_timeout_orders(self):
        """检查超时订单"""
        timed_out = []
        for order in self.active_orders:
            if order.is_overdue(self.current_time) and order.status != OrderStatus.COMPLETED:
                order.mark_timeout()
                timed_out.append(order)
                self.total_orders_timed_out += 1

        # 从活跃订单中移除超时订单
        self.active_orders = [order for order in self.active_orders if order not in timed_out]
        self.timed_out_orders.extend(timed_out)

    def _calculate_order_reward(self):
        """
        计算订单相关的奖励，返回 (total_reward, completion_revenue, timeout_penalty)。

        - completion_revenue: 纯配送完成收益（>= 0），不含超时惩罚
        - timeout_penalty: 订单超时惩罚（<= 0）
        - total_reward = (completion_revenue + timeout_penalty) * order_weight

        Returns:
            (total_reward, completion_revenue, timeout_penalty) 均已乘 order_weight
        """
        if not self.enable_orders:
            return 0.0, 0.0, 0.0

        completion_revenue = 0.0
        timeout_penalty = 0.0

        # 1. 订单完成收益：检测新完成的订单
        current_completed_count = self.total_orders_completed
        new_completions = current_completed_count - self.previous_completed_count
        if new_completions > 0:
            completion_reward = self.reward_config.get("order_completion_reward", 5.0)
            completion_revenue += new_completions * completion_reward
            self.previous_completed_count = current_completed_count

        # 2. 订单超时惩罚：检测新超时的订单
        current_timed_out_count = self.total_orders_timed_out
        new_timeouts = current_timed_out_count - self.previous_timed_out_count
        if new_timeouts > 0:
            pen = self.reward_config.get("order_timeout_penalty", -5.0)
            timeout_penalty += new_timeouts * pen
            self.previous_timed_out_count = current_timed_out_count

        order_weight = self.reward_config.get("order_weight", 1.0)
        return (completion_revenue + timeout_penalty) * order_weight, \
               completion_revenue * order_weight, \
               timeout_penalty * order_weight

    def _calculate_shaping_reward(self, uav_id: int, previous_position: Tuple[float, float], current_position: Tuple[float, float]) -> float:
        """
        计算 shaping reward：鼓励UAV朝向目标移动
        
        r_shape = k * (d_t - d_{t+1})
        - d_t: 上一步位置到目标点的距离
        - d_{t+1}: 当前步位置到目标点的距离
        - k: shaping_reward_coeff 缩放系数
        
        Args:
            uav_id: UAV ID
            previous_position: 移动前的位置 (lat, lng)
            current_position: 移动后的位置 (lat, lng)
        
        Returns:
            shaping reward 值
        """
        if not self.enable_orders:
            return 0.0

        task_manager = self.uav_task_managers[uav_id]
        current_target = task_manager.get_current_target()

        # 如果没有目标（IDLE或CHARGING状态），不给予shaping reward
        if current_target is None:
            # 清空该UAV的距离记录
            if uav_id in self.previous_target_distances:
                del self.previous_target_distances[uav_id]
            if uav_id in self.previous_targets:
                del self.previous_targets[uav_id]
            return 0.0

        # 检查目标是否改变（如果改变，重置距离记录）
        previous_target = self.previous_targets.get(uav_id)
        if previous_target != current_target:
            # 目标改变：基于 previous_position 计算距离并记录
            # 这样下一步计算时就能正确使用 previous_position 的距离作为 d_t
            # 使用米为单位计算距离
            previous_distance_to_new_target = euclidean_distance_meters(
                previous_position[0], previous_position[1],
                current_target[0], current_target[1]
            )
            self.previous_target_distances[uav_id] = previous_distance_to_new_target
            self.previous_targets[uav_id] = current_target
            return 0.0  # 目标改变时不计算shaping reward

        # 计算当前步到目标的距离 d_{t+1}（米）
        current_distance = euclidean_distance_meters(
            current_position[0], current_position[1],
            current_target[0], current_target[1]
        )

        # 如果上一步没有距离记录（首次计算），需要基于 previous_position 计算 d_t
        if uav_id not in self.previous_target_distances:
            # 计算 previous_position 到目标的距离作为 d_t（米）
            previous_distance = euclidean_distance_meters(
                previous_position[0], previous_position[1],
                current_target[0], current_target[1]
            )
            # 记录 current_distance 供下一步使用
            self.previous_target_distances[uav_id] = current_distance
            self.previous_targets[uav_id] = current_target
            # 计算距离变化：d_t - d_{t+1}
            distance_change = previous_distance - current_distance
        else:
            # 计算距离变化：d_t - d_{t+1}
            previous_distance = self.previous_target_distances[uav_id]
            distance_change = previous_distance - current_distance

        # 计算 shaping reward: r_shape = k * (d_t - d_{t+1})
        k = self.reward_config.get("shaping_reward_coeff", 1000.0)
        shaping_reward = k * distance_change

        # 更新距离记录供下一步使用
        self.previous_target_distances[uav_id] = current_distance
        self.previous_targets[uav_id] = current_target

        return shaping_reward

    def _calculate_sensing_coverage(self, lat: float, lng: float, trigger_events: List[str] = None) -> float:
        """
        计算感知覆盖奖励 φ(S') - 层次化熵基目标函数

        使用n小时窗口的层次化熵来评估感知覆盖质量，
        同时考虑数据平衡性和数据总量。

        Args:
            lat: UAV纬度
            lng: UAV经度
            trigger_events: 触发事件列表，决定是否收集传感数据

        Returns:
            感知覆盖奖励值 φ(A) = αE(A) + (1-α)log2(Q(A))
        """
        # 检查是否应该收集传感数据
        if self.sensing_reward_system.should_collect_at_position(lat, lng, self.current_time, trigger_events, self.sensing_config):
            self.sensing_reward_system.collect_sensor_data(lat, lng, self.current_time)

        # 计算当前感知覆盖奖励（基于所有已收集的传感数据）
        reward = self.sensing_reward_system.calculate_current_reward()

        return reward

    def extract_order_states(self, info):
        """
        从环境info中提取订单状态，用于Transformer编码

        Args:
            info: step()或reset()返回的info字典

        Returns:
            order_states: torch.Tensor, shape [num_orders, order_feature_dim]
                         如果没有订单，返回空tensor [0, order_feature_dim]
        """
        active_orders = info.get('active_orders', [])
        pending_orders = info.get('pending_orders', [])

        if not active_orders and not pending_orders:
            # 返回空tensor，保持维度一致性
            return torch.zeros(0, 8, dtype=torch.float32)

        order_features = []

        # 处理活跃订单和待处理订单
        all_orders = active_orders + pending_orders

        for order in all_orders:
            if hasattr(order, 'start_location') and hasattr(order, 'end_location'):
                # 标准Order对象
                features = [
                    float(order.start_location[0]),  # pickup_lat
                    float(order.start_location[1]),  # pickup_lng
                    float(order.end_location[0]),    # dropoff_lat
                    float(order.end_location[1]),    # dropoff_lng
                    order.time_window[0].timestamp() if hasattr(order, 'time_window') else 0.0,  # start_time
                    order.time_window[1].timestamp() if hasattr(order, 'time_window') else 0.0,  # end_time
                    getattr(order, 'priority', 1.0),  # priority
                    float(list(OrderStatus).index(getattr(order, 'status', OrderStatus.PENDING)))  # status
                ]
            else:
                # 如果是其他格式的订单数据，使用默认值
                features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

            order_features.append(features)

        return torch.tensor(order_features, dtype=torch.float32)

    def extract_weather_state(self, uav_states):
        """
        从UAV状态中提取天气信息（所有UAV共享相同天气）

        Args:
            uav_states: numpy array, shape [num_uavs, uav_feature_dim]
                       或单个UAV状态向量

        Returns:
            weather_state: torch.Tensor, shape [weather_feature_dim]
        """
        if uav_states.ndim == 1:
            # 单个UAV状态
            weather_features = uav_states[3:6]  # wind_speed, wind_direction, precipitation
        else:
            # 多个UAV状态，取第一个UAV的天气信息
            weather_features = uav_states[0, 3:6]

        return torch.tensor(weather_features, dtype=torch.float32)

    def get_enhanced_state(self, state, info, state_fusion_model=None):
        """
        获取增强的状态表示（使用Transformer融合）

        Args:
            state: 原始状态向量 (numpy array)
            info: 环境信息字典
            state_fusion_model: StateFusion模型实例，如果为None则返回原始状态

        Returns:
            enhanced_state: torch.Tensor 或 numpy array
        """
        if state_fusion_model is None:
            return state

        # 重新组织UAV状态
        uav_states = state.reshape(self.num_uavs, -1)  # [num_uavs, uav_feature_dim]
        uav_states = torch.tensor(uav_states, dtype=torch.float32).unsqueeze(0)  # [1, num_uavs, uav_feature_dim]

        # 提取订单状态
        order_states = self.extract_order_states(info).unsqueeze(0)  # [1, num_orders, order_feature_dim]

        # 提取天气状态
        weather_state = self.extract_weather_state(state).unsqueeze(0)  # [1, weather_feature_dim]

        # 使用StateFusion模型
        with torch.no_grad():
            enhanced_state = state_fusion_model(uav_states, order_states, weather_state)

        return enhanced_state.squeeze(0)  # [d_model]

    def render(self, mode='human'):
        """渲染环境（可选）"""
        # TODO: 用户可以实现可视化
        pass

    def close(self):
        """清理环境"""
        pass
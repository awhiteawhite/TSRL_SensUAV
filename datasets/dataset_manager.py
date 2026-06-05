"""
Dataset Manager for UAV Delivery System

This module handles loading and processing of real-world dataset files,
integrating with the configuration system for UAV path planning reinforcement learning.

"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class Order:
    """订单数据结构"""
    order_id: str
    region_id: str
    delivery_lat: float
    delivery_lng: float
    region_center_lat: float
    region_center_lng: float
    arrival_time: datetime
    ready_time: datetime  # 可开始服务时间
    due_time: datetime    # 必须完成时间
    demand: float = 1.0  # 需求量（可以是重量、体积等）


@dataclass
class RegionBounds:
    """区域边界"""
    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """转换为四元组格式，与配置文件兼容"""
        return (self.min_lat, self.max_lat, self.min_lng, self.max_lng)


@dataclass
class EpisodeData:
    """单局训练数据"""
    orders: List[Order]
    region_bounds: RegionBounds
    depot_location: Tuple[float, float]
    time_window: Tuple[datetime, datetime]
    # 可选：该episode对应的日期字符串（YYYY-MM-DD），对于daily模式很有用
    date_str: Optional[str] = None


class DatasetManager:
    """
    数据集管理器

    负责：
    - 加载和预处理CSV数据集
    - 计算区域边界
    - 生成训练episode
    - 与配置系统集成
    - 支持按天数据加载（用于贪心baseline评估）
    - 支持随机采样和按日期采样两种模式

    新增功能（用于贪心baseline）：
    - get_available_dates(): 获取所有可用日期
    - load_daily_data(): 加载指定日期的预处理数据
    - sample_episode_by_date(): 按指定日期采样episode
    - sample_all_daily_episodes(): 获取所有天的episodes
    """

    def __init__(self, csv_path: str, config_template: str = "default"):
        """
        初始化数据集管理器

        Args:
            csv_path: CSV文件路径
            config_template: 使用的配置模板名称
        """
        self.csv_path = Path(csv_path)
        self.config_template = config_template

        # 加载配置
        import sys
        import os
        # 添加项目根目录到Python路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from config import get_config_template
        self.config = get_config_template(config_template)

        # 加载和预处理数据
        self._load_data()
        self._compute_region_bounds()
        self._preprocess_orders()

    def _load_data(self):
        """加载CSV数据"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.csv_path}")

        try:
            self.raw_data = pd.read_csv(self.csv_path)
            print(f"成功加载数据集: {len(self.raw_data)} 条记录")
            print(f"列名: {list(self.raw_data.columns)}")
        except Exception as e:
            raise ValueError(f"读取CSV文件失败: {e}")

        # 检查必需的列（支持列名映射）
        column_mapping = {
            'pickup_lat': ['pickup_lat', 'accept_gps_lat'],
            'pickup_lng': ['pickup_lng', 'accept_gps_lng'],
            'arrival_time': ['arrival_time', 'accept_time']
        }

        required_columns = ['order_id', 'region_id']
        for logical_col, possible_cols in column_mapping.items():
            found = False
            for col in possible_cols:
                if col in self.raw_data.columns:
                    # 如果是映射的列名，重命名
                    if col != logical_col:
                        self.raw_data[logical_col] = self.raw_data[col]
                        print(f"column name mapping: {col} -> {logical_col}")
                    found = True
                    break
            if not found:
                required_columns.append(logical_col)

        missing_columns = [col for col in required_columns if col not in self.raw_data.columns and col not in column_mapping.keys()]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")

    def _compute_region_bounds(self):
        """根据region_id计算每个区域的边界"""
        self.region_bounds = {}

        # 按region_id分组计算边界
        # 按region_id分组计算边界
        for region_id, group in self.raw_data.groupby('region_id'):
            # 同时考虑pickup（accept）和delivery位置，取两者的并集
            region_id = str(region_id)
            all_lats = pd.concat([group['pickup_lat'], group['delivery_gps_lat']])
            all_lngs = pd.concat([group['pickup_lng'], group['delivery_gps_lng']])
            
            min_lat = all_lats.min()
            max_lat = all_lats.max()
            min_lng = all_lngs.min()
            max_lng = all_lngs.max()

            # 添加一些缓冲区（可选）
            lat_buffer = (max_lat - min_lat) * 0.1
            lng_buffer = (max_lng - min_lng) * 0.1

            self.region_bounds[region_id] = RegionBounds(
                min_lat=min_lat - lat_buffer,
                max_lat=max_lat + lat_buffer,
                min_lng=min_lng - lng_buffer,
                max_lng=max_lng + lng_buffer
            )

        print(f"computed {len(self.region_bounds)} region bounds")

    def _preprocess_orders(self):
        """预处理订单数据"""
        self.all_orders = []

        for _, row in self.raw_data.iterrows():
            # 解析到达时间
            if isinstance(row['arrival_time'], str):
                arrival_time = pd.to_datetime(row['arrival_time'])
            else:
                # 如果是数值型时间戳或其他格式，需要相应转换
                arrival_time = pd.to_datetime(row['arrival_time'], unit='s')  # 假设是秒级时间戳

            # 根据配置文件设置时间窗
            time_window_config = self.config.get('task_config', {}).get('time_window_minutes', 60)
            ready_time = arrival_time
            due_time = arrival_time + timedelta(minutes=time_window_config)

            order = Order(
                order_id=str(row['order_id']),
                region_id=str(row['region_id']),
                delivery_lat=float(row['delivery_gps_lat']),
                delivery_lng=float(row['delivery_gps_lng']),
                region_center_lat=float(row.get('region_center_lat')),
                region_center_lng=float(row.get('region_center_lng')),
                arrival_time=arrival_time.to_pydatetime(),
                ready_time=ready_time.to_pydatetime(),
                due_time=due_time.to_pydatetime(),
                demand=float(row.get('demand', 1.0))
            )

            self.all_orders.append(order)

        print(f"预处理了 {len(self.all_orders)} 个订单")

    def get_region_bounds(self, region_id: str) -> RegionBounds:
        """获取指定区域的边界"""
        if region_id not in self.region_bounds:
            raise ValueError(f"Unknown region ID: {region_id}")
        return self.region_bounds[region_id]

    def get_all_region_ids(self) -> List[str]:
        """获取所有区域ID"""
        return list(self.region_bounds.keys())

    def sample_orders_by_region(self, region_id: str, num_orders: int,
                               time_window: Optional[Tuple[datetime, datetime]] = None,
                               seed: Optional[int] = None) -> List[Order]:
        """
        从指定区域采样订单

        Args:
            region_id: 区域ID
            num_orders: 采样订单数量
            time_window: 时间窗口 (start, end)，如果为None则使用所有时间
            seed: 随机种子

        Returns:
            采样得到的订单列表
        """
        if seed is not None:
            np.random.seed(seed)

        # 筛选指定区域的订单
        region_orders = [order for order in self.all_orders if order.region_id == region_id]

        if not region_orders:
            print(f"Warning: region {region_id} has no order data")
            return []

        # 筛选时间窗口内的订单（如果指定了时间窗口）
        if time_window:
            start_time, end_time = time_window
            region_orders = [
                order for order in region_orders
                if start_time <= order.arrival_time <= end_time
            ]

        if len(region_orders) == 0:
            print(f"Warning: no orders in region {region_id} in the specified time window")
            return []

        # 采样订单
        if num_orders >= len(region_orders):
            sampled_orders = region_orders
        else:
            sampled_orders = np.random.choice(region_orders, size=num_orders, replace=False).tolist()

        return sampled_orders

    def get_available_dates(self, region_id: str) -> List[str]:
        """
        获取指定区域所有可用的日期列表。
        当 daily/ 中的文件包含多个城市数据时，只返回含有该 region_id 行的日期。
        结果按 region_id 缓存，避免训练中反复读取 92 个 CSV 文件。

        Args:
            region_id: 区域ID

        Returns:
            日期字符串列表，格式为YYYY-MM-DD
        """
        cache_key = str(region_id)
        if not hasattr(self, '_available_dates_cache'):
            self._available_dates_cache: Dict[str, List[str]] = {}
        if cache_key in self._available_dates_cache:
            return self._available_dates_cache[cache_key]

        daily_dir = self.csv_path.parent / 'daily'
        if not daily_dir.exists():
            print(f"Warning: daily directory does not exist {daily_dir}, please run preprocess_dataset.py first")
            return []

        region_id_int = int(region_id)
        date_files = []
        for file_path in daily_dir.glob('*.csv'):
            try:
                date_str = file_path.stem
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue

            try:
                df = pd.read_csv(file_path, usecols=['region_id'])
                if (df['region_id'].astype(int) == region_id_int).any():
                    date_files.append(date_str)
            except Exception:
                date_files.append(date_str)

        date_files.sort()
        self._available_dates_cache[cache_key] = date_files
        return date_files

    def split_train_test_dates(self, region_id: str, 
                              train_ratio: float = 0.8,
                              split_date: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        将可用日期分为训练集和测试集
        
        Args:
            region_id: 区域ID
            train_ratio: 训练集比例（如果split_date为None时使用）
            split_date: 指定的分割日期（YYYY-MM-DD），该日期及之前为训练集
        
        Returns:
            (train_dates, test_dates): 训练集和测试集的日期列表
        """
        available_dates = self.get_available_dates(region_id)
        if not available_dates:
            return [], []
        
        available_dates.sort()
        
        if split_date:
            # 按指定日期分割
            if split_date in available_dates:
                split_idx = available_dates.index(split_date) + 1
            else:
                # 找到最接近的日期
                split_idx = 0
                for i, date in enumerate(available_dates):
                    if date <= split_date:
                        split_idx = i + 1
                    else:
                        break
        else:
            # 按比例分割
            split_idx = int(len(available_dates) * train_ratio)
        
        train_dates = available_dates[:split_idx]
        test_dates = available_dates[split_idx:]
        
        return train_dates, test_dates

    def split_train_val_test_dates(
        self,
        region_id: str,
        train_ratio: float = 0.78,
        val_ratio: float = 0.11,
        test_ratio: float = 0.11,
        mode: str = "random",
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        将可用日期分为训练集、验证集、测试集（7:1:1）

        Args:
            region_id: 区域ID
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            mode: "chronological"（按时间顺序）或 "random"（随机打乱）
            seed: 随机种子（仅 random 模式使用）

        Returns:
            (train_dates, val_dates, test_dates)
        """
        available_dates = self.get_available_dates(region_id)
        if not available_dates:
            return [], [], []

        available_dates.sort()
        n = len(available_dates)

        total = train_ratio + val_ratio + test_ratio
        n_val = max(1, round(n * val_ratio / total))
        n_test = max(1, round(n * test_ratio / total))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            leftover = n - 1
            n_val = leftover // 2
            n_test = leftover - n_val

        if mode == "random":
            rng = np.random.RandomState(seed)
            indices = rng.permutation(n)
            train_idx = sorted(indices[:n_train])
            val_idx = sorted(indices[n_train:n_train + n_val])
            test_idx = sorted(indices[n_train + n_val:])
            train_dates = [available_dates[i] for i in train_idx]
            val_dates = [available_dates[i] for i in val_idx]
            test_dates = [available_dates[i] for i in test_idx]
        else:
            train_dates = available_dates[:n_train]
            val_dates = available_dates[n_train:n_train + n_val]
            test_dates = available_dates[n_train + n_val:]

        return train_dates, val_dates, test_dates

    @staticmethod
    def save_split(split_path: str, train_dates: List[str],
                   val_dates: List[str], test_dates: List[str],
                   meta: Optional[Dict] = None):
        """将划分结果保存为 JSON，确保所有实验使用同一份划分"""
        data = {
            "train_dates": train_dates,
            "val_dates": val_dates,
            "test_dates": test_dates,
            "n_train": len(train_dates),
            "n_val": len(val_dates),
            "n_test": len(test_dates),
        }
        if meta:
            data["meta"] = meta
        os.makedirs(os.path.dirname(os.path.abspath(split_path)) or ".", exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DatasetManager] Split saved to {split_path}: "
              f"train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)}")

    @staticmethod
    def load_split(split_path: str) -> Tuple[List[str], List[str], List[str]]:
        """从 JSON 加载已有的划分结果"""
        with open(split_path, "r") as f:
            data = json.load(f)
        train_dates = data["train_dates"]
        val_dates = data["val_dates"]
        test_dates = data["test_dates"]
        print(f"[DatasetManager] Split loaded from {split_path}: "
              f"train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)}")
        return train_dates, val_dates, test_dates

    @staticmethod
    def get_split_dates(
        dataset_manager,
        region_id: str,
        split_path: str = "datasets/data_split.json",
        train_ratio: float = 0.78,
        val_ratio: float = 0.11,
        test_ratio: float = 0.11,
        mode: str = "random",
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        便捷接口：如果 split_path 存在则加载，否则创建并保存。
        保证所有脚本使用同一份划分。
        """
        import os as _os
        if _os.path.exists(split_path):
            return DatasetManager.load_split(split_path)
        train_dates, val_dates, test_dates = dataset_manager.split_train_val_test_dates(
            region_id=region_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            mode=mode,
            seed=seed,
        )
        DatasetManager.save_split(
            split_path, train_dates, val_dates, test_dates,
            meta={"mode": mode, "seed": seed,
                  "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio},
        )
        return train_dates, val_dates, test_dates

    def load_daily_data(self, date_str: str, region_id: Optional[int] = None) -> pd.DataFrame:
        """
        加载指定日期的预处理数据

        Args:
            date_str: 日期字符串，格式为YYYY-MM-DD
            region_id: 区域ID过滤（可选）

        Returns:
            pandas DataFrame包含当天的订单数据
        """
        daily_dir = self.csv_path.parent / 'daily'
        daily_file = daily_dir / f"{date_str}.csv"

        if not daily_file.exists():
            raise FileNotFoundError(f"Date file not found: {daily_file}")

        try:
            df = pd.read_csv(daily_file)
            print(f"Loaded date data {date_str}: {len(df)} records")

            # 如果指定了region_id，则进行过滤（统一使用整数比较，避免字符串/整数类型不一致）
            if region_id is not None:
                original_count = len(df)
                # 将region_id列转换为整数以确保类型一致
                df['region_id'] = df['region_id'].astype(int)
                df = df[df['region_id'] == int(region_id)]
                print(f"Region filtered {region_id}: {original_count} -> {len(df)} records")

            return df

        except Exception as e:
            raise ValueError(f"Failed to load date file {daily_file}: {e}")

    def sample_episode_by_date(self, date_str: str, region_id: str,
                              num_orders: Optional[int] = None) -> EpisodeData:
        """
        按指定日期采样一个episode

        Args:
            date_str: 日期字符串，格式为YYYY-MM-DD
            region_id: 区域ID
            num_orders: 订单数量，None表示使用当天所有订单

        Returns:
            EpisodeData对象
        """
        # 加载当天数据
        daily_df = self.load_daily_data(date_str, region_id)

        if daily_df.empty:
            raise ValueError(f"Date {date_str} region {region_id} has no order data")

        # 获取区域边界
        region_bounds = self.get_region_bounds(region_id)

        # 设置仓库位置
        depot_lat = (region_bounds.min_lat + region_bounds.max_lat) / 2
        depot_lng = (region_bounds.min_lng + region_bounds.max_lng) / 2
        depot_location = (depot_lat, depot_lng)

        # 从配置中读取仓库位置（如果有的话）
        if 'depot_location' in self.config.get('uav_env_config', {}):
            depot_location = tuple(self.config['uav_env_config']['depot_location'])

        # 解析时间窗口（从配置读取开始时间，默认7点，结束时间为当天24:00）
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            # 从配置读取episode开始时间，默认7点
            episode_start_hour = self.config.get('uav_env_config', {}).get('episode_start_hour', 7)
            start_time = datetime.combine(date_obj, datetime.min.time().replace(hour=episode_start_hour))
            # 结束时间为当天24:00（即第二天的00:00，但用当天的日期表示）
            end_time = datetime.combine(date_obj, datetime.min.time().replace(hour=0)) + timedelta(days=1)
        except ValueError as e:
            raise ValueError(f"Invalid date format {date_str}: {e}")

        # 将DataFrame转换为Order对象，并过滤7点之前的订单
        orders = []
        for _, row in daily_df.iterrows():
            try:
                arrival_time = datetime.strptime(row['accept_time'], '%Y/%m/%d %H:%M')
                
                # 只保留在时间窗口内的订单（从start_time开始）
                if arrival_time < start_time:
                    continue
                    
                order = Order(
                    order_id=str(row['order_id']),
                    region_id=str(row['region_id']),
                    delivery_lat=float(row['delivery_gps_lat']),
                    delivery_lng=float(row['delivery_gps_lng']),
                    region_center_lat=float(row.get('region_center_lat')),
                    region_center_lng=float(row.get('region_center_lng')),
                    arrival_time=arrival_time,
                    ready_time=arrival_time,
                    due_time=datetime.strptime(row['delivery_time'], '%Y/%m/%d %H:%M'),
                    demand=1.0
                )
                orders.append(order)
            except (KeyError, ValueError) as e:
                print(f"Warning: skipping invalid order {row.get('order_id', 'unknown')}: {e}")
                continue

        # 如果指定了订单数量，进行采样
        if num_orders is not None and len(orders) > num_orders:
            np.random.seed(42)  # 使用固定种子确保可重复性
            orders = np.random.choice(orders, size=num_orders, replace=False).tolist()

        return EpisodeData(
            orders=orders,
            region_bounds=region_bounds,
            depot_location=depot_location,
            time_window=(start_time, end_time),
            date_str=date_str
        )

    def sample_all_daily_episodes(self, region_id: str,
                                 num_orders_per_episode: Optional[int] = None) -> Dict[str, EpisodeData]:
        """
        获取按日期分组的所有episodes（用于贪心baseline评估）

        Args:
            region_id: 区域ID
            num_orders_per_episode: 每个episode的订单数量，None表示使用当天所有订单

        Returns:
            字典：date_str -> EpisodeData
        """
        available_dates = self.get_available_dates(region_id)
        if not available_dates:
            print(f"Warning: region {region_id} has no available date data")
            return {}

        episodes = {}
        for date_str in available_dates:
            try:
                episode = self.sample_episode_by_date(
                    date_str=date_str,
                    region_id=region_id,
                    num_orders=num_orders_per_episode
                )
                episodes[date_str] = episode
                print(f"Successfully loaded episode: {date_str} - {len(episode.orders)} orders")
            except Exception as e:
                print(f"Warning: skipping date {date_str}: {e}")
                continue

        print(f"\nLoaded {len(episodes)} episodes, covering {len(available_dates)} available dates")
        return episodes

    def preload_episodes(self, dates: List[str], region_id: str,
                         num_orders: Optional[int] = None) -> Dict[str, 'EpisodeData']:
        """
        将指定日期列表的所有 EpisodeData 一次性加载到内存。
        训练开始前调用一次即可，后续 env.reset() 直接从内存取数据，
        不再有磁盘 I/O。

        Returns:
            {date_str: EpisodeData}，跳过加载失败的日期
        """
        episodes: Dict[str, EpisodeData] = {}
        for date_str in dates:
            try:
                ep = self.sample_episode_by_date(
                    date_str=date_str,
                    region_id=region_id,
                    num_orders=num_orders,
                )
                episodes[date_str] = ep
            except Exception as e:
                print(f"[preload] skipping {date_str}: {e}")
        print(f"[preload_episodes] {len(episodes)}/{len(dates)} dates loaded into memory")
        return episodes

    def sample_episode(self, region_id: str, num_orders: Optional[int],
                      episode_duration_hours: int = 8,
                      seed: Optional[int] = None,
                      mode: str = "random") -> EpisodeData:
        """
        采样一个训练episode

        Args:
            region_id: 区域ID
            num_orders: 订单数量
            episode_duration_hours: episode时长（小时）
            seed: 随机种子
            mode: 采样模式
                - "random": 原有随机采样逻辑（用于RL训练）
                - "daily": 随机选择一天进行采样（用于RL训练的daily模式）

        Returns:
            EpisodeData对象
        """
        if mode == "random":
            # 原有随机采样逻辑
            return self._sample_episode_random(region_id, num_orders, episode_duration_hours, seed)
        elif mode == "daily":
            # 随机选择一天进行采样（均匀随机）
            available_dates = self.get_available_dates(region_id)
            if not available_dates:
                raise ValueError(f"region {region_id} has no available date data")

            if seed is not None:
                np.random.seed(seed)
            selected_date = np.random.choice(available_dates)

            # 使用sample_episode_by_date来构建EpisodeData（其中包含date_str）
            return self.sample_episode_by_date(
                date_str=selected_date,
                region_id=region_id,
                num_orders=num_orders
            )
        else:
            raise ValueError(f"Unsupported sampling mode: {mode}")

    def _sample_episode_random(self, region_id: str, num_orders: int,
                              episode_duration_hours: int, seed: Optional[int]) -> EpisodeData:
        """
        原始的随机采样逻辑（私有方法）
        """
        if seed is not None:
            np.random.seed(seed)

        # 随机选择一个时间起点
        all_times = [order.arrival_time for order in self.all_orders if order.region_id == region_id]
        if not all_times:
            raise ValueError(f"region {region_id} has no order data")

        start_time = np.random.choice(all_times)
        end_time = start_time + timedelta(hours=episode_duration_hours)

        # 采样订单
        orders = self.sample_orders_by_region(
            region_id=region_id,
            num_orders=num_orders,
            time_window=(start_time, end_time),
            seed=seed
        )

        # 获取区域边界
        region_bounds = self.get_region_bounds(region_id)

        # 设置仓库位置
        depot_lat = (region_bounds.min_lat + region_bounds.max_lat) / 2
        depot_lng = (region_bounds.min_lng + region_bounds.max_lng) / 2
        depot_location = (depot_lat, depot_lng)

        # 从配置中读取仓库位置
        if 'depot_location' in self.config.get('uav_env_config', {}):
            depot_location = tuple(self.config['uav_env_config']['depot_location'])

        return EpisodeData(
            orders=orders,
            region_bounds=region_bounds,
            depot_location=depot_location,
            time_window=(start_time, end_time)
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_orders': len(self.all_orders),
            'total_regions': len(self.region_bounds),
            'region_stats': {}
        }

        for region_id, bounds in self.region_bounds.items():
            region_orders = [o for o in self.all_orders if o.region_id == region_id]
            region_stats = {
                'order_count': len(region_orders),
                'bounds': bounds.to_tuple(),
            }

            # 处理时间范围（避免空序列错误）
            if region_orders:
                region_stats['time_range'] = {
                    'min': min(o.arrival_time for o in region_orders).isoformat(),
                    'max': max(o.arrival_time for o in region_orders).isoformat()
                }
            else:
                region_stats['time_range'] = None

            stats['region_stats'][region_id] = region_stats

        return stats

    def save_statistics(self, output_path: str):
        """保存统计信息到文件"""
        stats = self.get_statistics()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Statistics saved to: {output_path}")


def create_config_from_dataset(csv_path: str, region_id: str,
                              output_config_path: str,
                              template: str = "default"):
    """
    根据数据集创建一个定制的配置文件

    Args:
        csv_path: CSV数据集路径
        region_id: 目标区域ID
        output_config_path: 输出配置文件路径
        template: 基础配置模板
    """
    # 加载数据集
    manager = DatasetManager(csv_path, template)

    # 获取区域边界
    bounds = manager.get_region_bounds(region_id)

    # 加载基础配置
    from config import get_config_template
    base_config = get_config_template(template, use_json=False)  # 使用Python字典版本

    # 更新配置
    custom_config = base_config.copy()
    custom_config.update({
        'region_bounds': bounds.to_tuple(),
        'depot_location': manager.sample_episode(region_id, 1).depot_location,
        'region_id': region_id,
        'dataset_path': str(csv_path)
    })

    # 保存配置
    import json
    with open(output_config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'description': f'Configuration based on dataset {Path(csv_path).name} region {region_id}',
            'region_id': region_id,
            'uav_env_config': {
                'region_bounds': list(bounds.to_tuple()),
                'depot_location': list(custom_config['depot_location']),
                **base_config
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"Custom configuration saved to: {output_config_path}")
    return custom_config


# 测试和使用示例
if __name__ == "__main__":
    # 简单测试新功能
    import os
    script_dir = Path(__file__).parent
    csv_file = script_dir / "hangzhou_region0_101_MayToJuly.csv"

    if not csv_file.exists():
        print(f"Dataset file not found: {csv_file}")
        exit(1)

    try:
        # 创建数据集管理器
        print("Initializing DatasetManager...")
        manager = DatasetManager(str(csv_file), config_template="debug")
        print("Initialization completed")

        # 测试新功能：按天数据加载
        print("\nTesting daily data loading functionality:")

        # 获取所有可用区域
        region_ids = manager.get_all_region_ids()
        print(f"Available regions: {region_ids}")

        if region_ids:
            region_id = region_ids[0]  # 使用第一个区域
            print(f"Using region: {region_id}")

            # 获取可用日期
            available_dates = manager.get_available_dates(region_id)
            print(f"Available dates: {len(available_dates)}")
            if available_dates:
                print(f"Date range: {available_dates[0]} to {available_dates[-1]}")

                # 测试加载指定日期的数据
                test_date = available_dates[0]
                print(f"\nTesting loading data for date {test_date}...")
                daily_df = manager.load_daily_data(test_date, region_id)
                print(f"Successfully loaded: {len(daily_df)} records")

                # 测试按日期采样episode
                print(f"\nTesting sampling episode by date...")
                episode = manager.sample_episode_by_date(
                    date_str=test_date,
                    region_id=region_id,
                    num_orders=5  # 少量订单用于快速测试
                )
                print(f"Sampling successful: {len(episode.orders)} orders")

                print("\n[OK] All new functionality tests passed!")
                print("Now you can use these functionalities for greedy baseline evaluation")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

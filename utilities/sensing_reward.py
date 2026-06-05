import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta


class SpatioTemporalWindowManager:
    """
    时空窗口管理器：维护n小时的滑动时间窗口和空间网格映射
    """

    def __init__(self, window_hours: int, grid_rows: int, grid_cols: int,
                 region_bounds: Tuple[float, float, float, float]):
        """
        初始化时空窗口管理器

        Args:
            window_hours: 窗口时长（小时）
            grid_rows: 空间网格行数
            grid_cols: 空间网格列数
            region_bounds: 区域边界 (min_lat, max_lat, min_lng, max_lng)
        """
        self.window_hours = window_hours
        self.window_steps = window_hours * 60  # 每小时60个steps
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.region_bounds = region_bounds

        # 预计算网格映射参数
        self.lat_range = region_bounds[1] - region_bounds[0]
        self.lng_range = region_bounds[3] - region_bounds[2]
        self.lat_step = self.lat_range / grid_rows
        self.lng_step = self.lng_range / grid_cols

        # 时间窗口状态
        self.window_data = {}  # {(i,j,t): count} 的稀疏存储
        self.current_time_idx = 0
        self.total_samples = 0

        # 过期时间戳管理（用于滑动窗口）
        self.time_stamps = []

    def lat_lng_to_grid(self, lat: float, lng: float) -> Tuple[int, int]:
        """
        将经纬度坐标转换为网格坐标

        Args:
            lat: 纬度
            lng: 经度

        Returns:
            (row_idx, col_idx): 网格坐标
        """
        # 边界检查和裁剪
        lat = np.clip(lat, self.region_bounds[0], self.region_bounds[1])
        lng = np.clip(lng, self.region_bounds[2], self.region_bounds[3])

        # 计算网格索引
        row_idx = int((lat - self.region_bounds[0]) / self.lat_step)
        col_idx = int((lng - self.region_bounds[2]) / self.lng_step)

        # 确保索引在有效范围内
        row_idx = min(max(row_idx, 0), self.grid_rows - 1)
        col_idx = min(max(col_idx, 0), self.grid_cols - 1)

        return row_idx, col_idx

    def add_sample(self, lat: float, lng: float, timestamp: datetime) -> Tuple[int, int, int]:
        """
        添加一个数据样本到窗口中（episode 内全局累积，不做滑动窗口淘汰）。

        时间索引 t 为自首个样本以来的分钟数，可超过 window_steps；
        所有样本在 episode 内始终保留，直到 reset() 清空。

        Args:
            lat: 纬度
            lng: 经度
            timestamp: 时间戳

        Returns:
            (i, j, t): 网格坐标和相对时间索引
        """
        i, j = self.lat_lng_to_grid(lat, lng)

        if not self.time_stamps:
            self.time_stamps.append(timestamp)
            t = 0
        else:
            time_diff = (timestamp - self.time_stamps[0]).total_seconds() / 60
            t = int(time_diff)

        key = (i, j, t)
        if key not in self.window_data:
            self.window_data[key] = 0
        self.window_data[key] += 1
        self.total_samples += 1

        return i, j, t

    def _slide_window(self, steps_to_slide: int):
        """
        滑动时间窗口，移除过期数据

        Args:
            steps_to_slide: 需要滑动的步数
        """
        if steps_to_slide <= 0:
            return

        # 更新基准时间
        if self.time_stamps:
            self.time_stamps[0] += timedelta(minutes=steps_to_slide)

        # 移除过期数据
        keys_to_remove = []
        for key in self.window_data:
            i, j, t = key
            if t < steps_to_slide:
                keys_to_remove.append(key)
                self.total_samples -= self.window_data[key]

        for key in keys_to_remove:
            del self.window_data[key]

        # 更新所有剩余数据的相对时间索引
        new_window_data = {}
        for key, count in self.window_data.items():
            i, j, t = key
            new_t = t - steps_to_slide
            if new_t >= 0:
                new_window_data[(i, j, new_t)] = count

        self.window_data = new_window_data

    def get_data_at_granularity(self, k: int) -> Dict[Tuple[int, int, int], int]:
        """
        获取粒度k下的数据分布

        Args:
            k: 粒度级别 (1=细粒度, 2=中粒度, etc.)

        Returns:
            {(i,j,t): count} 在粒度k下的数据分布
        """
        granularity_data = {}

        for (i, j, t), count in self.window_data.items():
            # 根据粒度k聚合空间和时间
            i_k = i // k
            j_k = j // k
            t_k = t // k

            key = (i_k, j_k, t_k)
            if key not in granularity_data:
                granularity_data[key] = 0
            granularity_data[key] += count

        return granularity_data

    def get_current_stats(self) -> Dict:
        """
        获取当前窗口的统计信息

        Returns:
            包含各种统计信息的字典
        """
        return {
            'total_samples': self.total_samples,
            'unique_grids': len(self.window_data),
            'window_hours': self.window_hours,
            'grid_shape': (self.grid_rows, self.grid_cols),
            'time_span': len(self.time_stamps) if self.time_stamps else 0
        }


class HierarchicalEntropyCalculator:
    """
    层次化熵计算器：实现基于Appendix 1的O(1)高效更新算法
    """

    def __init__(self, window_manager: SpatioTemporalWindowManager, kmax: int = 3):
        """
        初始化层次化熵计算器

        Args:
            window_manager: 时空窗口管理器实例
            kmax: 最大粒度级别
        """
        self.window_manager = window_manager
        self.kmax = kmax

        # 为每个粒度维护状态
        self.Q = 0.0  # 总数据量
        self.E = {k: 0.0 for k in range(1, kmax + 1)}  # 各粒度的熵值
        self.sum_log_terms = {k: {} for k in range(1, kmax + 1)}  # Σ A*log2(A) 项

        # 权重因子缓存
        self.weights = self._calculate_weights()

    def _calculate_weights(self) -> Dict[int, float]:
        """
        计算各粒度的权重因子 ω(k) = log2(I(1)J(1)T(1)) / log2(I(k)J(k)T(k))

        Returns:
            各粒度的权重因子
        """
        weights = {}

        # 基准粒度k=1的维度
        I1 = self.window_manager.grid_rows
        J1 = self.window_manager.grid_cols
        T1 = self.window_manager.window_steps

        max_entropy_1 = math.log2(I1 * J1 * T1) if I1 * J1 * T1 > 1 else 1.0

        for k in range(1, self.kmax + 1):
            Ik = I1 // k
            Jk = J1 // k
            Tk = T1 // k

            max_entropy_k = math.log2(Ik * Jk * Tk) if Ik * Jk * Tk > 1 else 1.0

            if max_entropy_k > 0:
                weights[k] = max_entropy_1 / max_entropy_k
            else:
                weights[k] = 1.0

        return weights

    def update_entropy(self, i: int, j: int, t: int) -> float:
        """
        使用O(1)算法更新熵值（基于Appendix 1）

        Args:
            i, j, t: 新增数据的位置和时间

        Returns:
            更新后的层次化熵值 E(A)
        """
        # 更新总数据量
        old_Q = self.Q
        self.Q = old_Q + 1

        # 为每个粒度更新熵值
        for k in range(1, self.kmax + 1):
            self._update_entropy_at_granularity(k, i, j, t, old_Q)

        # 计算层次化熵 E(A) = Σ ω(k)E(A(k))/kmax
        hierarchical_entropy = sum(self.weights[k] * self.E[k] for k in range(1, self.kmax + 1)) / self.kmax

        return hierarchical_entropy

    def _update_entropy_at_granularity(self, k: int, i: int, j: int, t: int, old_Q: float):
        """
        更新特定粒度k的熵值

        Args:
            k: 粒度级别
            i, j, t: 新增数据的原始坐标
            old_Q: 更新前的总数据量
        """
        # 计算粒度k下的坐标
        ik = i // k
        jk = j // k
        tk = t // k

        # 获取当前位置的当前计数
        key = (ik, jk, tk)
        current_count = self.sum_log_terms[k].get(key, 0)

        # 计算新计数
        new_count = current_count + 1

        # 使用Appendix 1的公式更新熵
        if old_Q > 0:
            # 从公式：E(A(k)) = log2(Q+1) - 1/(Q+1) * {Q*(log2(Q) - E_old) - old_count*log2(old_count)} - 1/(Q+1)*(new_count)*log2(new_count)

            log_old_Q = math.log2(old_Q)
            E_old = self.E[k]

            # 计算括号内的项
            term1 = old_Q * (log_old_Q - E_old)
            term2 = current_count * math.log2(current_count) if current_count > 0 else 0
            term3 = new_count * math.log2(new_count)

            # 更新熵值
            log_new_Q = math.log2(self.Q)
            self.E[k] = log_new_Q - (1.0 / self.Q) * (term1 - term2) - (1.0 / self.Q) * term3
        else:
            # 初始情况
            self.E[k] = math.log2(new_count) if new_count > 1 else 0.0

        # 更新sum_log_terms
        self.sum_log_terms[k][key] = new_count

    def resync_from_window_data(self) -> None:
        """
        从 window_manager.window_data 完整重建熵计算器状态。

        在 _slide_window 删除过期样本后必须调用，以确保 Q / E[k] / sum_log_terms
        与窗口内实际留存的样本保持一致，消除增量更新与滑动窗口之间的状态不同步问题。

        时间复杂度：O(U × kmax)，U 为 window_data 中唯一格子数。
        在 sensing 任务稀疏（≤20次/episode）的场景下开销可忽略不计。
        """
        window_data = self.window_manager.window_data

        # 对每个粒度 k 重新聚合粗格子计数
        new_sum_log_terms: Dict[int, Dict] = {k: {} for k in range(1, self.kmax + 1)}
        for (i, j, t), count in window_data.items():
            for k in range(1, self.kmax + 1):
                ck = (i // k, j // k, t // k)
                new_sum_log_terms[k][ck] = new_sum_log_terms[k].get(ck, 0) + count
        self.sum_log_terms = new_sum_log_terms

        # 重建总样本数 Q
        new_Q = float(sum(window_data.values()))
        self.Q = new_Q

        # 重建各粒度的 Shannon 熵 E[k] = log2(Q) - (1/Q) * Σ n * log2(n)
        if new_Q > 0:
            log_Q = math.log2(new_Q)
            for k in range(1, self.kmax + 1):
                sum_n_log_n = sum(
                    n * math.log2(n)
                    for n in self.sum_log_terms[k].values()
                    if n > 0
                )
                self.E[k] = log_Q - sum_n_log_n / new_Q
        else:
            self.E = {k: 0.0 for k in range(1, self.kmax + 1)}

    def get_current_entropy(self) -> float:
        """
        获取当前的层次化熵值

        Returns:
            层次化熵值 E(A)
        """
        if self.Q == 0:
            return 0.0

        hierarchical_entropy = sum(self.weights[k] * self.E[k] for k in range(1, self.kmax + 1)) / self.kmax
        return hierarchical_entropy

    def get_data_quantity(self) -> float:
        """
        获取当前的数据总量

        Returns:
            数据总量 Q(A)
        """
        return self.Q


class SensingRewardInterface:
    """
    感知奖励接口：与uav_environment.py对接的统一接口

    负责管理全局传感数据的收集和层次化熵的计算
    """

    def __init__(self, window_hours: int = 2, grid_rows: int = 20, grid_cols: int = 20,
                 region_bounds: Tuple[float, float, float, float] = None, alpha: float = 0.7):
        """
        初始化感知奖励接口

        Args:
            window_hours: 时间窗口（小时）
            grid_rows: 空间网格行数
            grid_cols: 空间网格列数
            region_bounds: 区域边界 (min_lat, max_lat, min_lng, max_lng)
            alpha: 平衡参数 α ∈ [0,1]，平衡数据平衡和数据总量的重要性
        """
        self.alpha = alpha

        # 默认区域边界（如果未提供）
        if region_bounds is None:
            region_bounds = (-100, 100, -100, 100)

        # 初始化组件
        self.window_manager = SpatioTemporalWindowManager(
            window_hours=window_hours,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            region_bounds=region_bounds
        )

        self.entropy_calculator = HierarchicalEntropyCalculator(
            window_manager=self.window_manager,
            kmax=3  # 使用3个粒度级别
        )

        # 传感数据收集管理
        self.collected_sensor_data = []  # 收集到的传感数据点 [(lat, lng, timestamp), ...]

    def collect_sensor_data(self, lat: float, lng: float, timestamp: datetime) -> None:
        """
        在指定位置收集传感数据

        Args:
            lat: 传感数据纬度
            lng: 传感数据经度
            timestamp: 收集时间戳
        """
        # 记录传感数据点
        self.collected_sensor_data.append((lat, lng, timestamp))

        # 添加到时空窗口并用 O(1) 增量更新熵（不再有滑动窗口，无需 resync）
        i, j, t = self.window_manager.add_sample(lat, lng, timestamp)
        self.entropy_calculator.update_entropy(i, j, t)

    def calculate_current_reward(self) -> float:
        """
        计算当前的感知覆盖奖励（基于已收集的所有传感数据）

        Returns:
            感知覆盖奖励值 φ(A) = αE(A) + (1-α)log2(Q(A))
        """
        # 获取当前层次化熵和数据总量
        hierarchical_entropy = self.entropy_calculator.get_current_entropy()
        data_quantity = self.entropy_calculator.get_data_quantity()

        # 计算最终奖励 φ(A) = αE(A) + (1-α)log2(Q(A))
        if data_quantity > 0:
            log_Q = math.log2(data_quantity)
            reward = self.alpha * hierarchical_entropy + (1 - self.alpha) * log_Q
        else:
            reward = 0.0

        return reward

    def should_collect_at_position(self, uav_lat: float, uav_lng: float, timestamp: datetime,
                                  trigger_events: List[str] = None, sensing_config: Dict = None) -> bool:
        """
        判断是否应该在当前位置收集传感数据

        Args:
            uav_lat: UAV纬度
            uav_lng: UAV经度
            timestamp: 当前时间戳
            trigger_events: 触发事件列表 (如 ['order_pickup', 'order_delivery', 'periodic'])
            sensing_config: 感知配置参数

        Returns:
            是否应该收集传感数据
        """
        if trigger_events is None:
            trigger_events = []

        if sensing_config is None:
            sensing_config = {}

        # 检查各种触发条件
        should_collect = False

        # 1. 订单相关事件触发
        if ('order_pickup' in trigger_events and sensing_config.get('collect_on_pickup', True)) or \
           ('order_delivery' in trigger_events and sensing_config.get('collect_on_delivery', True)):
            should_collect = True

        # 2. 定期收集（每隔一定时间）
        if sensing_config.get('periodic_collection', False):
            minutes_since_last = 0
            if self.collected_sensor_data:
                last_timestamp = self.collected_sensor_data[-1][2]
                minutes_since_last = (timestamp - last_timestamp).total_seconds() / 60

            periodic_interval = sensing_config.get('periodic_interval_minutes', 5)
            if minutes_since_last >= periodic_interval:
                should_collect = True

        # 3. 随机收集（配置概率）
        random_prob = sensing_config.get('random_collection_prob', 0.0)
        if random_prob > 0 and np.random.random() < random_prob:
            should_collect = True

        return should_collect

    def update_region_bounds(self, new_bounds: Tuple[float, float, float, float]):
        """
        更新区域边界（用于动态调整）

        Args:
            new_bounds: 新的区域边界
        """
        self.window_manager.region_bounds = new_bounds
        # 重新计算网格映射参数
        self.window_manager.lat_range = new_bounds[1] - new_bounds[0]
        self.window_manager.lng_range = new_bounds[3] - new_bounds[2]
        self.window_manager.lat_step = self.window_manager.lat_range / self.window_manager.grid_rows
        self.window_manager.lng_step = self.window_manager.lng_range / self.window_manager.grid_cols

    def get_stats(self) -> Dict:
        """
        获取系统统计信息

        Returns:
            统计信息字典
        """
        window_stats = self.window_manager.get_current_stats()
        entropy_stats = {
            'current_entropy': self.entropy_calculator.get_current_entropy(),
            'data_quantity': self.entropy_calculator.get_data_quantity(),
            'alpha': self.alpha
        }

        return {**window_stats, **entropy_stats}

    def reset(self):
        """
        重置系统状态（用于新episode）
        """
        self.window_manager.window_data = {}
        self.window_manager.current_time_idx = 0
        self.window_manager.total_samples = 0
        self.window_manager.time_stamps = []

        self.entropy_calculator.Q = 0.0
        self.entropy_calculator.E = {k: 0.0 for k in range(1, self.entropy_calculator.kmax + 1)}
        self.entropy_calculator.sum_log_terms = {k: {} for k in range(1, self.entropy_calculator.kmax + 1)}


# 测试函数
def test_sensing_reward_system():
    """测试感知奖励系统的基本功能"""
    import datetime

    # 创建测试实例
    system = SensingRewardInterface(
        window_hours=2,
        grid_rows=10,
        grid_cols=10,
        region_bounds=(-100, 100, -100, 100),
        alpha=0.7
    )

    # 测试基本功能
    base_time = datetime.datetime(2022, 5, 1, 12, 0, 0)

    rewards = []
    for i in range(10):
        lat = -80 + i * 20  # -80到+100
        lng = -80 + i * 20  # -80到+100
        timestamp = base_time + datetime.timedelta(minutes=i)

        reward = system.calculate_reward(lat, lng, timestamp)
        rewards.append(reward)
        print(".3f")

    # 检查统计信息
    stats = system.get_stats()
    print(f"\nFinal stats: {stats}")

    # 验证奖励单调递增（因为数据量在增加）
    assert all(rewards[i] <= rewards[i+1] for i in range(len(rewards)-1)), "Rewards should be non-decreasing"
    print("✓ Test passed: Rewards are non-decreasing")

    return True


if __name__ == "__main__":
    test_sensing_reward_system()

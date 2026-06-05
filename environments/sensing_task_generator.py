"""
机会性感知任务生成器

生成随机分布在region内的感知点，UAV经过这些点时自动收集传感数据。
这些点在每个episode开始时生成一次，episode内保持不变。
"""

import random
from typing import List, Tuple, Set
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_manager import RegionBounds
from utilities.geo_utils import euclidean_distance_meters


class SensingTaskGenerator:
    """
    机会性感知任务生成器

    在region边界内随机生成固定数量的感知点，这些点在整个episode内保持不变。
    UAV经过这些点时自动触发数据收集。
    """

    def __init__(self, region_bounds: RegionBounds, num_points: int, detection_threshold: float = 10.0):
        """
        初始化感知任务生成器

        Args:
            region_bounds: 区域边界对象
            num_points: 感知点数量
            detection_threshold: 检测阈值（米）
        """
        self.region_bounds = region_bounds
        self.num_points = num_points
        self.detection_threshold = detection_threshold

        # 感知点列表 [(lat, lng), ...]
        self.sensing_points: List[Tuple[float, float]] = []

        # 已访问的点集合（使用索引避免重复）
        self.visited_points: Set[int] = set()

        # 当前step新检测到的点（用于批量处理）
        self.newly_detected_points: Set[int] = set()

        # 生成感知点
        self._generate_sensing_points()

    def _generate_sensing_points(self):
        """
        在region边界内随机生成感知点
        episode开始时调用一次，episode内不再改变
        """
        min_lat, max_lat, min_lng, max_lng = self.region_bounds.to_tuple()

        self.sensing_points = []
        for _ in range(self.num_points):
            # 在region范围内随机生成坐标
            lat = random.uniform(min_lat, max_lat)
            lng = random.uniform(min_lng, max_lng)
            self.sensing_points.append((lat, lng))

        # 重置已访问集合
        self.visited_points.clear()

        print(f"Generated {self.num_points} opportunistic sensing points in region")

    def check_and_collect(self, uav_position: Tuple[float, float]) -> bool:
        """
        检查UAV当前位置是否可以进行机会性感知
        注意：这个方法只记录新检测到的点，不立即处理数据收集

        Args:
            uav_position: UAV当前位置 (lat, lng)

        Returns:
            bool: 是否检测到新的感知点
        """
        uav_lat, uav_lng = uav_position

        for i, point in enumerate(self.sensing_points):
            if i in self.visited_points:
                continue  # 已访问过，跳过

            point_lat, point_lng = point

            # 使用米为单位计算距离
            distance = euclidean_distance_meters(uav_lat, uav_lng, point_lat, point_lng)

            if distance <= self.detection_threshold:
                # 发现可感知点，记录为新检测到的点
                self.newly_detected_points.add(i)
                return True  # 检测成功

        return False  # 无检测到新点

    def process_collected_data(self, sensing_system, timestamp) -> int:
        """
        处理当前step中所有新检测到的感知点
        对每个新检测到的点调用sensing_system.collect_sensor_data

        Args:
            sensing_system: 感知奖励系统实例
            timestamp: 当前时间戳

        Returns:
            int: 处理的感知点数量
        """
        processed_count = 0

        for point_idx in self.newly_detected_points:
            # 确保这个点还没有被处理过
            if point_idx not in self.visited_points:
                # 获取感知点坐标
                point_lat, point_lng = self.sensing_points[point_idx]

                # 调用感知系统收集数据
                sensing_system.collect_sensor_data(point_lat, point_lng, timestamp)

                # 标记为已访问
                self.visited_points.add(point_idx)
                processed_count += 1

        # 清除新检测到的点列表，为下一个step做准备
        self.newly_detected_points.clear()

        return processed_count

    def get_sensing_points(self) -> List[Tuple[float, float]]:
        """
        获取所有感知点坐标

        Returns:
            感知点列表 [(lat, lng), ...]
        """
        return self.sensing_points.copy()

    def get_visited_count(self) -> int:
        """
        获取已访问的感知点数量

        Returns:
            已访问点数量
        """
        return len(self.visited_points)

    def get_total_points(self) -> int:
        """
        获取总感知点数量

        Returns:
            总点数量
        """
        return self.num_points

    def get_visit_rate(self) -> float:
        """
        获取访问率

        Returns:
            访问率 (0.0-1.0)
        """
        if self.num_points == 0:
            return 0.0
        return len(self.visited_points) / self.num_points

    def reset_for_new_episode(self):
        """
        为新episode重置（重新生成感知点）
        """
        self._generate_sensing_points()
        self.newly_detected_points.clear()

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        return {
            'total_points': self.num_points,
            'visited_points': len(self.visited_points),
            'newly_detected_points': len(self.newly_detected_points),
            'visit_rate': self.get_visit_rate(),
            'region_bounds': self.region_bounds.to_tuple(),
            'detection_threshold': self.detection_threshold
        }


# 辅助函数
def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    计算两点间的距离（米）

    Args:
        point1: 点1坐标 (lat, lng)
        point2: 点2坐标 (lat, lng)

    Returns:
        距离（米）
    """
    lat1, lng1 = point1
    lat2, lng2 = point2
    return euclidean_distance_meters(lat1, lng1, lat2, lng2)
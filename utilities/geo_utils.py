"""
地理坐标工具函数
提供经纬度与米之间的距离转换
"""

import numpy as np
from typing import Tuple

# 地球平均半径（米）
EARTH_RADIUS_METERS = 6371000.0

# 1度纬度对应的米数（近似固定值）
METERS_PER_LAT_DEGREE = 111320.0


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    使用Haversine公式计算两个经纬度点之间的实际距离（米）
    
    Args:
        lat1: 第一个点的纬度
        lng1: 第一个点的经度
        lat2: 第二个点的纬度
        lng2: 第二个点的经度
    
    Returns:
        两点之间的距离（米）
    """
    # 转换为弧度
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lng = np.radians(lng2 - lng1)
    
    # Haversine公式
    a = np.sin(delta_lat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return EARTH_RADIUS_METERS * c


def euclidean_distance_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    使用简化的欧几里得近似计算两点间距离（米）
    适用于小范围区域（几十公里内），计算速度更快
    
    Args:
        lat1: 第一个点的纬度
        lng1: 第一个点的经度
        lat2: 第二个点的纬度
        lng2: 第二个点的经度
    
    Returns:
        两点之间的近似距离（米）
    """
    # 计算平均纬度，用于经度的修正
    avg_lat = (lat1 + lat2) / 2
    
    # 纬度差转米（1度纬度 ≈ 111320米）
    delta_lat_meters = (lat2 - lat1) * METERS_PER_LAT_DEGREE
    
    # 经度差转米（需要乘以cos(纬度)修正）
    delta_lng_meters = (lng2 - lng1) * METERS_PER_LAT_DEGREE * np.cos(np.radians(avg_lat))
    
    return np.sqrt(delta_lat_meters ** 2 + delta_lng_meters ** 2)


def meters_to_lat_degrees(meters: float) -> float:
    """
    将米转换为纬度度数
    
    Args:
        meters: 距离（米）
    
    Returns:
        对应的纬度度数
    """
    return meters / METERS_PER_LAT_DEGREE


def meters_to_lng_degrees(meters: float, latitude: float) -> float:
    """
    将米转换为经度度数（需要考虑纬度）
    
    Args:
        meters: 距离（米）
        latitude: 当前纬度（用于修正）
    
    Returns:
        对应的经度度数
    """
    return meters / (METERS_PER_LAT_DEGREE * np.cos(np.radians(latitude)))


def lat_degrees_to_meters(degrees: float) -> float:
    """
    将纬度度数转换为米
    
    Args:
        degrees: 纬度度数
    
    Returns:
        对应的米数
    """
    return degrees * METERS_PER_LAT_DEGREE


def lng_degrees_to_meters(degrees: float, latitude: float) -> float:
    """
    将经度度数转换为米（需要考虑纬度）
    
    Args:
        degrees: 经度度数
        latitude: 当前纬度（用于修正）
    
    Returns:
        对应的米数
    """
    return degrees * METERS_PER_LAT_DEGREE * np.cos(np.radians(latitude))


def delta_meters_to_lat_lng(delta_x_meters: float, delta_y_meters: float, 
                            current_lat: float) -> Tuple[float, float]:
    """
    将米为单位的位移转换为经纬度位移
    
    假设 delta_x 对应纬度方向（南北），delta_y 对应经度方向（东西）
    
    Args:
        delta_x_meters: 纬度方向位移（米），正值向北
        delta_y_meters: 经度方向位移（米），正值向东
        current_lat: 当前纬度（用于经度修正）
    
    Returns:
        (delta_lat, delta_lng) 经纬度位移
    """
    delta_lat = meters_to_lat_degrees(delta_x_meters)
    delta_lng = meters_to_lng_degrees(delta_y_meters, current_lat)
    return delta_lat, delta_lng


def lat_lng_delta_to_meters(delta_lat: float, delta_lng: float, 
                            current_lat: float) -> Tuple[float, float]:
    """
    将经纬度位移转换为米为单位的位移
    
    Args:
        delta_lat: 纬度位移
        delta_lng: 经度位移
        current_lat: 当前纬度（用于经度修正）
    
    Returns:
        (delta_x_meters, delta_y_meters) 米为单位的位移
    """
    delta_x_meters = lat_degrees_to_meters(delta_lat)
    delta_y_meters = lng_degrees_to_meters(delta_lng, current_lat)
    return delta_x_meters, delta_y_meters


# 为了向后兼容，提供一个简单的别名
calculate_distance_meters = euclidean_distance_meters


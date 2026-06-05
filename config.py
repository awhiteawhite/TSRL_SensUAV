import json
import os
from pathlib import Path

# =============================================================================
# 环境参数 (Environment Parameters)
# =============================================================================

# UAV 环境参数
UAV_ENV_CONFIG = {
    # 区域边界：(min_lat, max_lat, min_lng, max_lng)
    "region_bounds": (-100, 100, -100, 100),

    # 最大步数（单episode）
    "max_steps": 1000,

    # 低电量阈值（当电量低于此值时强制返回仓库）
    "low_battery_threshold": 0.3,

    # 仓库位置：(lat, lng)
    "depot_location": (0.0, 0.0),

    # UAV数量
    "num_uavs": 1,

    # 订单系统参数
    "enable_orders": True,

    # 订单时间窗参数（分钟）
    "order_time_window_min": 30,
    "order_time_window_max": 50,

    # ==================== 速度控制相关配置 ====================
    # 最大速度 (m/min)
    "max_speed": 500.0,

    # 最小速度 (m/min)
    "min_speed": 0.0,

    # 近目标减速距离阈值 (m)：当距离目标小于此值时强制减速
    "slow_down_distance": 300.0,

    # 近目标减速后的速度 (m/min)
    "slow_down_speed": 20.0,

    # ==================== 能量消耗相关配置 ====================
    # 能量消耗系数：E = energy_coeff * v_eff² * t，v_eff = v + w_tailwind（沿航向风分量）
    "energy_coeff": 0.000001,

    # 电池容量（用于归一化电量消耗）
    "battery_capacity": 10000.0,
}

# =============================================================================
# 数据集参数 (Dataset Parameters)
# =============================================================================

DATASET_CONFIG = {
    # 数据集路径
    "dataset_path": "datasets",

    # 数据文件名
    "orders_filename": "hangzhou_region0_101_MayToJuly.csv",
    "weather_filename": "hourly_data_region101_hz_May_June_July_2022.csv",

    # 区域ID
    "region_id": 0,

    # 是否使用真实数据集
    "use_real_datasets": True,
}

# =============================================================================
# 感知覆盖参数 (Sensing Coverage Parameters)
# =============================================================================

SENSING_CONFIG = {
    # 时间窗口参数
    "window_hours": 2,

    # 空间网格参数
    "grid_rows": 20,
    "grid_cols": 20,

    # 平衡参数 α (数据平衡 vs 数据总量)
    "alpha": 0.5,

    # Preference sensing 点数量
    "num_sensing_points": 20,

    # UAV 到达 sensing 点的判定距离（米）
    "sensing_completion_distance": 100.0,

    # Sensing 点时间窗参数（分钟）
    "time_window_minutes_min": 600,
    "time_window_minutes_max": 900,

    # Sensing 点优先级范围
    "priority_range_min": 0.5,
    "priority_range_max": 1.0,

    # Sensing 超时惩罚（原始值，未乘 sensing_weight；由 PreferenceEnvWrapper 读取）
    "sensing_timeout_penalty": -5.0,
}

# =============================================================================
# 训练参数 (Training Parameters)
# =============================================================================

TRAINING_CONFIG = {
    # 实验名称
    "exp_name": "UAV_RPO",

    # 随机种子
    "seed": 1,

    # PyTorch 确定性设置
    "torch_deterministic": True,

    # 是否启用 CUDA
    "cuda": True,

    # 是否使用 Weights & Biases 跟踪
    "track": False,

    # W&B 项目名称
    "wandb_project_name": "UAV_PPO_RPO",

    # W&B 实体（团队）
    "wandb_entity": None,

    # 是否捕获视频
    "capture_video": False,

    # 环境ID
    "env_id": "UAV-v0",

    # 总训练步数
    "total_timesteps": 8000000,

    # 学习率
    "learning_rate": 3e-4,

    # 并行环境数量
    "num_envs": 1,

    # 每次策略 rollout 的步数
    "num_steps": 2048,

    # 是否启用学习率衰减
    "anneal_lr": True,

    # 折扣因子 gamma
    "gamma": 0.99,

    # GAE lambda 参数
    "gae_lambda": 0.95,

    # mini-batch 数量
    "num_minibatches": 32,

    # 每次更新的 epochs 数
    "update_epochs": 10,

    # 是否规范化优势函数
    "norm_adv": True,

    # PPO 裁剪系数
    "clip_coef": 0.2,

    # 是否裁剪价值函数损失
    "clip_vloss": True,

    # 熵系数
    "ent_coef": 0.0,

    # 价值函数系数
    "vf_coef": 0.5,

    # 梯度裁剪最大值
    "max_grad_norm": 0.5,

    # 目标 KL 散度阈值 (None 表示不使用)
    "target_kl": None,

    # RPO alpha 参数
    "rpo_alpha": 0.5
}

# =============================================================================
# DDPG 算法参数 (DDPG Algorithm Parameters)
# =============================================================================

DDPG_CONFIG = {
    # 学习率（Actor 和 Critic 使用相同学习率）
    "learning_rate": 3e-4,

    # Rollout Buffer 大小
    "buffer_size": int(1e6),  # 1,000,000

    # Batch size（从 Replay Buffer 采样）
    "batch_size": 256,

    # 折扣因子 gamma
    "gamma": 0.99,

    # Target network 软更新系数 tau
    "tau": 0.005,

    # 探索噪声系数（添加到确定性动作上）
    "exploration_noise": 0.1,

    # 开始学习前的步数（先收集经验）
    "learning_starts": 25000,

    # 策略更新频率（每 N 步更新一次 Actor）
    "policy_frequency": 2,

    # 是否保存模型
    "save_model": False,

    # 是否上传模型到 Hugging Face
    "upload_model": False,

    # Hugging Face 实体名
    "hf_entity": ""
}

# =============================================================================
# 奖励函数参数 (Reward Function Parameters)
# =============================================================================

REWARD_CONFIG = {
    # 下层 sensing 熵增量放大系数：由 UAVEnvironment 在 UAV 完成 sensing 采集时使用
    # 与上层 macro 的 sensing_weight（用于 timeout 惩罚加权）语义不同，请勿混淆
    "sensing_entropy_weight": 2000.0,

    # 天气惩罚权重
    "weather_weight": 1.0,

    # 电量惩罚权重
    "battery_penalty_weight": 1.0,

    # 返回仓库惩罚权重
    "return_penalty_weight": 1.0,

    # 边界惩罚权重
    "boundary_penalty_weight": 1.0,

    # 风力惩罚系数
    "wind_penalty_coeff": 0.1,

    # 降水惩罚系数
    "precipitation_penalty_coeff": 0.2,

    # 覆盖奖励的距离缩放因子
    "coverage_distance_scale": 1.0,

    # 订单完成奖励
    "order_completion_reward": 200.0,

    # 订单超时惩罚
    "order_timeout_penalty": -500.0,

    # 时间窗超时惩罚权重
    "time_window_penalty": -5.0,

    # 订单奖励权重
    "order_weight": 1.0,

    # 移动惩罚系数（基于移动距离，单位：米）- 已废弃
    "movement_penalty_coeff": 0.0001,

    # ==================== 能量惩罚配置 ====================
    # 能量惩罚系数：r_energy = -energy_penalty_coeff * v_eff² * t
    # v_eff = v + w_tailwind（沿航向风分量），逆风增大能耗、顺风减小能耗
    "energy_penalty_coeff": 0.0001,

    # Shaping reward 系数：鼓励UAV朝向目标移动
    # r_shape = k * (d_t - d_{t+1})，其中 d_t 是当前到目标距离（米），d_{t+1} 是下一步到目标距离（米）
    # 当 k=0.001 时，如果一步减少1000米距离，r_shape=1.0；如果减少2000米，r_shape=2.0
    "shaping_reward_coeff": 0.001
}

# =============================================================================
# 任务相关参数 (Task Parameters)
# =============================================================================

TASK_CONFIG = {
    # 是否启用任务系统
    "enable_tasks": True,

    # 任务数量
    "num_tasks": 10,

    # 任务位置随机范围
    "task_region_bounds": (-80, 80, -80, 80),

    # 任务完成距离阈值（米），UAV与任务点距离小于此值时视为任务完成
    "task_completion_distance": 10.0,

    # 任务完成奖励
    "task_completion_reward": 10.0,

    # 任务超时惩罚
    "task_timeout_penalty": -5.0,

    # 任务最大持续时间
    "max_task_duration": 100,

    # 订单系统参数
    "enable_orders": True,

    # 订单完成距离阈值（米），UAV与订单起点/终点距离小于此值时视为到达
    "order_completion_distance": 10.0,

    # 最大等待订单数量
    "max_pending_orders": 50,
    
}

# =============================================================================
# 气象参数 (Weather Parameters)
# =============================================================================

WEATHER_CONFIG = {
    # 是否使用真实气象数据
    "use_real_weather": False,

    # 风力参数（正态分布）
    "wind_mean": 0.0,
    "wind_std": 1.0,

    # 降水参数（指数分布）
    "precipitation_rate": 0.5,

    # 气象更新频率（每多少步更新一次）
    "weather_update_freq": 10,

    # 是否启用季节性气象变化
    "seasonal_weather": False,
}

# =============================================================================
# 日志和保存参数 (Logging and Saving Parameters)
# =============================================================================

LOGGING_CONFIG = {
    # TensorBoard 日志目录
    "tensorboard_log_dir": "runs",

    # 模型保存频率（每多少步保存一次）
    "save_freq": 100000,

    # 模型保存目录
    "model_save_dir": "models",

    # 是否保存最佳模型
    "save_best_model": True,

    # 评估频率
    "eval_freq": 50000,

    # 评估环境数量
    "eval_num_envs": 5,
}

# =============================================================================
# 预处理参数 (Preprocessing Parameters)
# =============================================================================

PREPROCESS_CONFIG = {
    # 观察空间裁剪范围
    "obs_clip_range": (-10, 10),

    # 奖励裁剪范围
    "reward_clip_range": (-10, 10),

    # 是否应用观察空间标准化
    "normalize_obs": True,

    # 是否应用奖励标准化
    "normalize_reward": True,

    # 是否裁剪动作
    "clip_actions": True,
}

# =============================================================================
# 调试参数 (Debug Parameters)
# =============================================================================

DEBUG_CONFIG = {
    # 调试模式
    "debug_mode": False,

    # 详细日志
    "verbose_logging": False,

    # 渲染环境
    "render_env": False,

    # 渲染频率
    "render_freq": 1000,

    # 打印频率
    "print_freq": 100,
}

# =============================================================================
# 便捷配置模板 (Configuration Templates)
# =============================================================================

def get_config_template(template_name="default", use_json=True):
    """
    获取预定义的配置模板

    Args:
        template_name: 模板名称
            - "default": 默认配置
            - "debug": 调试配置
            - "fast_training": 快速训练配置
            - "production": 生产环境配置
        use_json: 是否从JSON文件加载配置，默认True

    Returns:
        完整的配置字典
    """
    if use_json:
        # 从JSON文件加载配置
        json_filename = f"{template_name}.json"
        try:
            return load_config_from_json(json_filename)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            print(f"警告: 无法加载JSON配置 '{json_filename}'，回退到Python模板")
            use_json = False

    if not use_json:
        # 使用Python字典模板（原有逻辑）
        templates = {
            "default": {
                **UAV_ENV_CONFIG,
                **DATASET_CONFIG,
                **SENSING_CONFIG,
                **TRAINING_CONFIG,
                **DDPG_CONFIG,
                **REWARD_CONFIG,
                **TASK_CONFIG,
                **WEATHER_CONFIG,
                **LOGGING_CONFIG,
                **PREPROCESS_CONFIG,
                **DEBUG_CONFIG,
            },

            "debug": {
                **UAV_ENV_CONFIG,
                **DATASET_CONFIG,
                **SENSING_CONFIG,
                **TRAINING_CONFIG,
                **DDPG_CONFIG,
                **REWARD_CONFIG,
                **TASK_CONFIG,
                **WEATHER_CONFIG,
                **LOGGING_CONFIG,
                **PREPROCESS_CONFIG,
                **DEBUG_CONFIG,

                # 调试覆盖
                "total_timesteps": 10000,
                "num_steps": 256,
                "num_envs": 1,
                "num_minibatches": 4,
                "debug_mode": True,
                "verbose_logging": True,
                "print_freq": 10,
            },

            "fast_training": {
                **UAV_ENV_CONFIG,
                **DATASET_CONFIG,
                **SENSING_CONFIG,
                **TRAINING_CONFIG,
                **DDPG_CONFIG,
                **REWARD_CONFIG,
                **TASK_CONFIG,
                **WEATHER_CONFIG,
                **LOGGING_CONFIG,
                **PREPROCESS_CONFIG,
                **DEBUG_CONFIG,

                # 快速训练覆盖
                "total_timesteps": 100000,
                "num_steps": 512,
                "num_envs": 4,
                "learning_rate": 1e-3,
                "update_epochs": 5,
            },

            "production": {
                **UAV_ENV_CONFIG,
                **DATASET_CONFIG,
                **SENSING_CONFIG,
                **TRAINING_CONFIG,
                **DDPG_CONFIG,
                **REWARD_CONFIG,
                **TASK_CONFIG,
                **WEATHER_CONFIG,
                **LOGGING_CONFIG,
                **PREPROCESS_CONFIG,
                **DEBUG_CONFIG,

                # 生产环境覆盖
                "total_timesteps": 20000000,
                "num_envs": 8,
                "track": True,
                "save_freq": 50000,
                "eval_freq": 25000,
            },
        }

        return templates.get(template_name, templates["default"])

# =============================================================================
# 配置验证 (Configuration Validation)
# =============================================================================

def validate_config(config):
    """
    验证配置参数的合理性

    Args:
        config: 配置字典

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # 验证环境参数
    if config["low_battery_threshold"] < 0 or config["low_battery_threshold"] > 1:
        errors.append("low_battery_threshold 必须在 [0, 1] 范围内")

    if config["battery_consumption_rate"] <= 0:
        errors.append("battery_consumption_rate 必须大于 0")

    # 验证训练参数
    if config["num_envs"] <= 0:
        errors.append("num_envs 必须大于 0")

    if config["num_steps"] <= 0:
        errors.append("num_steps 必须大于 0")

    if config["learning_rate"] <= 0:
        errors.append("learning_rate 必须大于 0")

    # 验证奖励参数
    if config["rpo_alpha"] < 0:
        errors.append("rpo_alpha 不能为负数")

    return len(errors) == 0, errors

# =============================================================================
# 工具函数 (Utility Functions)
# =============================================================================

def load_config_from_json(json_path):
    """
    从JSON文件加载配置

    Args:
        json_path: JSON配置文件路径（相对路径或绝对路径）

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: JSON格式错误
        ValueError: 配置格式错误
    """
    # 转换为Path对象，支持相对路径
    config_path = Path(json_path)

    # 如果是相对路径，相对于configs目录
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / "configs" / json_path

    # 检查文件是否存在
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {e}")

    # 验证配置结构
    required_sections = [
        "uav_env_config", "training_config", "reward_config",
        "task_config", "weather_config", "logging_config",
        "preprocess_config", "debug_config"
    ]

    for section in required_sections:
        if section not in config_data:
            raise ValueError(f"配置缺少必需的部分: {section}")

    # 将嵌套配置展平为单一字典（与原有格式兼容）
    flattened_config = {}
    for section_name, section_data in config_data.items():
        if section_name == "description":
            continue  # 跳过描述字段
        if isinstance(section_data, dict):
            flattened_config.update(section_data)

    return flattened_config

def get_config_from_json(json_filename="default.json"):
    """
    从JSON文件获取配置（便捷函数）

    Args:
        json_filename: JSON配置文件名

    Returns:
        配置字典
    """
    try:
        return load_config_from_json(json_filename)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"警告: 无法加载JSON配置 '{json_filename}': {e}")
        print("将使用默认Python配置作为后备方案")
        return get_config_template("default")

def get_config_from_file(json_path):
    """
    从指定的JSON文件路径加载配置

    Args:
        json_path: JSON文件路径（绝对路径或相对于项目根目录的路径）

    Returns:
        配置字典

    Example:
        config = get_config_from_file("configs/my_custom_config.json")
        config = get_config_from_file("/absolute/path/to/config.json")
    """
    # 如果路径以"configs/"开头，移除它以避免重复
    if json_path.startswith("configs/"):
        json_path = json_path[8:]  # 移除"configs/"前缀

    try:
        return load_config_from_json(json_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"警告: 无法加载JSON配置 '{json_path}': {e}")
        print("将使用默认Python配置作为后备方案")
        return get_config_template("default")

def save_config(config, filepath):
    """
    保存配置到文件

    Args:
        config: 配置字典
        filepath: 保存路径
    """
    import json

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config(filepath):
    """
    从文件加载配置

    Args:
        filepath: 文件路径

    Returns:
        配置字典
    """
    import json

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_config(config, title="Configuration"):
    """
    打印配置信息

    Args:
        config: 配置字典
        title: 标题
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

    # 重新组织配置为分节显示
    section_mappings = {
        "UAV_ENV_CONFIG": ["enable_orders", "dataset_path", "region_bounds", "max_steps",
                          "low_battery_threshold", "depot_location", "battery_consumption_rate",
                          "max_movement", "num_uavs", "order_time_window_min", "order_time_window_max"],
        "DATASET_CONFIG": ["dataset_path", "orders_filename", "weather_filename", "region_id", "use_real_datasets"],
        "SENSING_CONFIG": ["window_hours", "grid_rows", "grid_cols", "alpha", "collect_on_pickup",
                          "collect_on_delivery", "periodic_collection", "periodic_interval_minutes", "random_collection_prob"],
        "TRAINING_CONFIG": ["exp_name", "seed", "torch_deterministic", "cuda", "track",
                           "wandb_project_name", "wandb_entity", "capture_video", "env_id",
                           "total_timesteps", "learning_rate", "num_envs", "num_steps",
                           "anneal_lr", "gamma", "gae_lambda", "num_minibatches",
                           "update_epochs", "norm_adv", "clip_coef", "clip_vloss",
                           "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "rpo_alpha"],
        "DDPG_CONFIG": ["learning_rate", "buffer_size", "batch_size", "gamma", "tau",
                       "exploration_noise", "learning_starts", "policy_frequency",
                       "save_model", "upload_model", "hf_entity"],
        "REWARD_CONFIG": ["sensing_weight", "weather_weight", "battery_penalty_weight",
                         "return_penalty_weight", "boundary_penalty_weight", "wind_penalty_coeff",
                         "precipitation_penalty_coeff", "coverage_distance_scale"],
        "TASK_CONFIG": ["enable_tasks", "num_tasks", "task_region_bounds",
                       "task_completion_distance", "task_completion_reward",
                       "task_timeout_penalty", "max_task_duration"],
        "WEATHER_CONFIG": ["use_real_weather", "wind_mean", "wind_std",
                          "precipitation_rate", "weather_update_freq", "seasonal_weather"],
        "LOGGING_CONFIG": ["tensorboard_log_dir", "save_freq", "model_save_dir",
                          "save_best_model", "eval_freq", "eval_num_envs"],
        "PREPROCESS_CONFIG": ["obs_clip_range", "reward_clip_range", "normalize_obs",
                             "normalize_reward", "clip_actions"],
        "DEBUG_CONFIG": ["debug_mode", "verbose_logging", "render_env",
                        "render_freq", "print_freq"]
    }

    for section_name, keys in section_mappings.items():
        print(f"\n{section_name}:")
        for key in keys:
            if key in config:
                print(f"  {key}: {config[key]}")

    print(f"\n{'='*50}\n")

# =============================================================================
# 默认配置 (Default Configuration)
# =============================================================================

# 默认使用JSON配置，如果JSON文件不存在则回退到Python模板
DEFAULT_CONFIG = get_config_template("default", use_json=True)

if __name__ == "__main__":
    # 测试配置加载
    print("测试JSON配置加载...")

    # 测试默认配置
    print("\n1. 测试默认配置:")
    config_default = get_config_from_json("default.json")
    is_valid, errors = validate_config(config_default)
    if is_valid:
        print("✅ 默认配置验证通过")
    else:
        print("❌ 默认配置验证失败:")
        for error in errors:
            print(f"  - {error}")

    # 测试调试配置
    print("\n2. 测试调试配置:")
    config_debug = get_config_from_json("debug.json")
    is_valid, errors = validate_config(config_debug)
    if is_valid:
        print("✅ 调试配置验证通过")
        print_config(config_debug, "调试配置")
    else:
        print("❌ 调试配置验证失败:")
        for error in errors:
            print(f"  - {error}")

    # 测试快速训练配置
    print("\n3. 测试快速训练配置:")
    config_fast = get_config_from_json("fast_training.json")
    is_valid, errors = validate_config(config_fast)
    if is_valid:
        print("✅ 快速训练配置验证通过")
    else:
        print("❌ 快速训练配置验证失败:")
        for error in errors:
            print(f"  - {error}")

    # 测试生产配置
    print("\n4. 测试生产配置:")
    config_prod = get_config_from_json("production.json")
    is_valid, errors = validate_config(config_prod)
    if is_valid:
        print("✅ 生产配置验证通过")
    else:
        print("❌ 生产配置验证失败:")
        for error in errors:
            print(f"  - {error}")

    # 测试自定义配置文件路径
    print("\n5. 测试自定义配置文件路径:")
    try:
        custom_config = get_config_from_file("configs/default.json")
        print("✅ 自定义路径配置加载成功")
    except Exception as e:
        print(f"❌ 自定义路径配置加载失败: {e}")

    print("\n配置系统测试完成!")

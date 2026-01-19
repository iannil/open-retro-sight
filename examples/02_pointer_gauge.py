#!/usr/bin/env python3
"""
Open-RetroSight 示例 02: 指针仪表读取

功能说明:
- 创建模拟指针仪表图像
- 配置仪表参数（量程、角度范围）
- 使用 PointerRecognizer 识别指针角度和数值
- 演示两点校准功能

依赖:
- opencv-python-headless
- numpy

运行方法:
    python examples/02_pointer_gauge.py
"""

import numpy as np
import cv2
import math
import tempfile


def create_gauge_image(pointer_angle: float = 45.0) -> np.ndarray:
    """
    创建模拟指针仪表图像

    Args:
        pointer_angle: 指针角度（度数，0度为3点钟方向，逆时针为正）

    Returns:
        指针仪表图像
    """
    # 创建白色背景
    size = 300
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    center = (size // 2, size // 2)
    radius = 120

    # 绘制表盘边框
    cv2.circle(img, center, radius, (0, 0, 0), 2)
    cv2.circle(img, center, radius + 5, (100, 100, 100), 1)

    # 绘制刻度
    for i in range(0, 360, 30):
        angle_rad = math.radians(i)
        # 刻度外端点
        x1 = int(center[0] + radius * math.cos(angle_rad))
        y1 = int(center[1] - radius * math.sin(angle_rad))
        # 刻度内端点
        inner_r = radius - 10 if i % 90 == 0 else radius - 5
        x2 = int(center[0] + inner_r * math.cos(angle_rad))
        y2 = int(center[1] - inner_r * math.sin(angle_rad))
        thickness = 2 if i % 90 == 0 else 1
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)

    # 绘制中心点
    cv2.circle(img, center, 8, (50, 50, 50), -1)

    # 绘制指针
    angle_rad = math.radians(pointer_angle)
    tip_x = int(center[0] + radius * 0.8 * math.cos(angle_rad))
    tip_y = int(center[1] - radius * 0.8 * math.sin(angle_rad))
    cv2.line(img, center, (tip_x, tip_y), (255, 0, 0), 3)

    # 绘制指针尾部
    tail_x = int(center[0] - 15 * math.cos(angle_rad))
    tail_y = int(center[1] + 15 * math.sin(angle_rad))
    cv2.line(img, center, (tail_x, tail_y), (255, 0, 0), 3)

    return img


def basic_pointer_example():
    """基础指针识别示例"""
    print("=" * 50)
    print("Open-RetroSight 指针仪表识别示例")
    print("=" * 50)

    from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

    # 创建仪表配置
    print("\n1. 配置仪表参数...")
    config = GaugeConfig(
        min_value=0,       # 最小值
        max_value=100,     # 最大值
        min_angle=225,     # 最小值对应角度（左下）
        max_angle=-45,     # 最大值对应角度（右下）
        unit="MPa"         # 单位
    )
    print(f"   量程: {config.min_value} - {config.max_value} {config.unit}")
    print(f"   角度范围: {config.min_angle}° - {config.max_angle}°")

    # 创建识别器
    print("\n2. 初始化 PointerRecognizer...")
    recognizer = PointerRecognizer(config)

    # 测试不同角度
    test_angles = [45, 90, 135, 180, 225]

    print("\n3. 识别结果:")
    print("-" * 60)
    print(f"{'设置角度':^12} | {'检测角度':^12} | {'数值':^12} | {'置信度':^12}")
    print("-" * 60)

    for angle in test_angles:
        # 创建测试图像
        image = create_gauge_image(angle)

        # 识别
        result = recognizer.recognize(image)

        # 输出
        detected_angle = f"{result.angle:.1f}°" if result.angle is not None else "N/A"
        value = f"{result.value:.1f}" if result.value is not None else "N/A"
        confidence = f"{result.confidence:.2%}"

        print(f"{angle:^12}° | {detected_angle:^12} | {value:^12} | {confidence:^12}")

    print("-" * 60)


def calibration_example():
    """校准示例"""
    print("\n" + "=" * 50)
    print("指针仪表校准示例")
    print("=" * 50)

    from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

    # 创建识别器
    recognizer = PointerRecognizer(GaugeConfig())

    print("\n1. 两点校准")
    print("   已知校准点:")
    print("   - 角度 45° 对应数值 25")
    print("   - 角度 135° 对应数值 75")

    # 执行两点校准
    recognizer.calibrate_two_point(
        angle1=45.0, value1=25.0,
        angle2=135.0, value2=75.0
    )
    print("   校准完成!")

    # 验证校准
    print("\n2. 验证校准结果:")
    test_angles = [45, 90, 135]

    for angle in test_angles:
        image = create_gauge_image(angle)
        result = recognizer.recognize(image)
        print(f"   角度 {angle}° -> 数值 {result.value:.1f}" if result.value else f"   角度 {angle}° -> N/A")


def save_load_calibration_example():
    """保存和加载校准示例"""
    print("\n" + "=" * 50)
    print("校准数据保存和加载示例")
    print("=" * 50)

    from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        calibration_file = f.name

    # 原始识别器 - 执行校准
    print("\n1. 创建识别器并校准...")
    recognizer1 = PointerRecognizer(GaugeConfig())
    recognizer1.calibrate_two_point(
        angle1=0.0, value1=0.0,
        angle2=180.0, value2=100.0
    )

    # 保存校准
    print(f"2. 保存校准到: {calibration_file}")
    recognizer1.save_calibration(calibration_file)

    # 新识别器 - 加载校准
    print("3. 创建新识别器并加载校准...")
    recognizer2 = PointerRecognizer(GaugeConfig())
    recognizer2.load_calibration(calibration_file)

    print("4. 验证加载的校准:")
    print(f"   校准数据有效: {recognizer2.calibration.is_valid()}")

    # 测试
    image = create_gauge_image(90)
    result = recognizer2.recognize(image)
    print(f"   90° 角度识别结果: {result.value:.1f}" if result.value else "   识别失败")

    import os
    os.unlink(calibration_file)
    print("\n   临时文件已清理")


def continuous_monitoring_example():
    """连续监控示例"""
    print("\n" + "=" * 50)
    print("连续监控示例（模拟）")
    print("=" * 50)

    from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig
    from retrosight.preprocessing.filter import create_default_filter

    recognizer = PointerRecognizer(GaugeConfig(
        min_value=0,
        max_value=100,
        min_angle=225,
        max_angle=-45
    ))

    # 创建滤波器
    value_filter = create_default_filter()

    print("\n模拟 10 帧连续读取:")
    print("-" * 50)
    print(f"{'帧':^6} | {'原始值':^12} | {'滤波值':^12} | {'置信度':^10}")
    print("-" * 50)

    # 模拟指针缓慢移动
    for i in range(10):
        # 模拟指针角度变化（45° -> 135°）
        angle = 45 + i * 10

        # 识别
        image = create_gauge_image(angle)
        result = recognizer.recognize(image)

        raw_value = result.value if result.value is not None else 0.0
        filtered_value = value_filter.filter(raw_value)

        print(f"{i+1:^6} | {raw_value:^12.1f} | {filtered_value:^12.1f} | {result.confidence:^10.2%}")

    print("-" * 50)


if __name__ == "__main__":
    # 基础示例
    basic_pointer_example()

    # 校准示例
    calibration_example()

    # 保存/加载校准
    save_load_calibration_example()

    # 连续监控
    continuous_monitoring_example()

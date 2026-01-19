#!/usr/bin/env python3
"""
Open-RetroSight 示例 03: 指示灯监控

功能说明:
- 创建模拟指示灯图像（绿/黄/红）
- 检测指示灯颜色和状态
- 演示 Andon 灯检测

依赖:
- opencv-python-headless
- numpy

运行方法:
    python examples/03_indicator_light.py
"""

import numpy as np
import cv2


def create_light_image(color: str = "green", is_on: bool = True) -> np.ndarray:
    """
    创建模拟指示灯图像

    Args:
        color: 灯颜色 ("green", "yellow", "red")
        is_on: 是否亮起

    Returns:
        指示灯图像
    """
    # 深灰色背景
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    # 灯座
    cv2.circle(img, (50, 50), 35, (30, 30, 30), -1)

    # 灯颜色 (BGR)
    color_map = {
        "green": (0, 255, 0) if is_on else (0, 80, 0),
        "yellow": (0, 255, 255) if is_on else (0, 80, 80),
        "red": (0, 0, 255) if is_on else (0, 0, 80),
        "off": (50, 50, 50)
    }

    light_color = color_map.get(color, color_map["off"])

    # 绘制灯
    cv2.circle(img, (50, 50), 30, light_color, -1)

    # 添加高光（如果亮起）
    if is_on and color != "off":
        cv2.circle(img, (40, 40), 8, (255, 255, 255), -1)

    return img


def create_andon_tower(states: dict) -> np.ndarray:
    """
    创建 Andon 灯塔图像

    Args:
        states: {"green": bool, "yellow": bool, "red": bool}

    Returns:
        Andon 灯塔图像
    """
    # 创建垂直排列的三色灯
    img = np.zeros((320, 100, 3), dtype=np.uint8)
    img[:] = (70, 70, 70)

    # 灯塔底座
    cv2.rectangle(img, (20, 300), (80, 320), (40, 40, 40), -1)
    cv2.rectangle(img, (30, 10), (70, 300), (60, 60, 60), -1)

    # 三色灯（从上到下：红、黄、绿）
    colors_positions = [
        ("red", 60),
        ("yellow", 150),
        ("green", 240)
    ]

    for color, y in colors_positions:
        is_on = states.get(color, False)
        color_map = {
            "green": (0, 255, 0) if is_on else (0, 60, 0),
            "yellow": (0, 255, 255) if is_on else (0, 60, 60),
            "red": (0, 0, 255) if is_on else (0, 0, 60)
        }
        cv2.circle(img, (50, y), 25, color_map[color], -1)
        if is_on:
            cv2.circle(img, (42, y - 8), 6, (255, 255, 255), -1)

    return img


def basic_light_detection():
    """基础指示灯检测示例"""
    print("=" * 50)
    print("Open-RetroSight 指示灯检测示例")
    print("=" * 50)

    from retrosight.recognition.light import LightRecognizer, LightConfig, LightColor

    # 配置
    print("\n1. 配置指示灯检测器...")
    config = LightConfig(
        region=(20, 20, 60, 60),  # 检测区域
        expected_colors=[LightColor.GREEN, LightColor.YELLOW, LightColor.RED]
    )

    recognizer = LightRecognizer(config)
    print("   检测器就绪")

    # 测试不同颜色的灯
    print("\n2. 检测不同颜色指示灯:")
    print("-" * 50)
    print(f"{'测试':^12} | {'检测颜色':^12} | {'状态':^12}")
    print("-" * 50)

    test_cases = [
        ("绿灯(亮)", "green", True),
        ("绿灯(灭)", "green", False),
        ("黄灯(亮)", "yellow", True),
        ("红灯(亮)", "red", True),
        ("红灯(灭)", "red", False),
    ]

    for name, color, is_on in test_cases:
        image = create_light_image(color, is_on)
        result = recognizer.detect(image)

        detected_color = str(result.color).split('.')[-1] if result.color else "N/A"
        state = str(result.state).split('.')[-1] if result.state else "N/A"

        print(f"{name:^12} | {detected_color:^12} | {state:^12}")

    print("-" * 50)


def andon_detection_example():
    """Andon 灯检测示例"""
    print("\n" + "=" * 50)
    print("Andon 灯塔检测示例")
    print("=" * 50)

    from retrosight.recognition.light import detect_andon

    # 不同的 Andon 状态
    andon_states = [
        {"name": "正常运行", "green": True, "yellow": False, "red": False},
        {"name": "警告", "green": False, "yellow": True, "red": False},
        {"name": "故障停机", "green": False, "yellow": False, "red": True},
        {"name": "待机", "green": False, "yellow": False, "red": False},
        {"name": "异常+警告", "green": False, "yellow": True, "red": True},
    ]

    print("\n检测结果:")
    print("-" * 60)
    print(f"{'状态名称':^15} | {'绿':^8} | {'黄':^8} | {'红':^8}")
    print("-" * 60)

    for state in andon_states:
        # 创建单色灯图像进行检测（简化示例）
        green_img = create_light_image("green", state["green"])
        result = detect_andon(green_img)

        # 显示配置的状态
        green_status = "ON" if state["green"] else "OFF"
        yellow_status = "ON" if state["yellow"] else "OFF"
        red_status = "ON" if state["red"] else "OFF"

        print(f"{state['name']:^15} | {green_status:^8} | {yellow_status:^8} | {red_status:^8}")

    print("-" * 60)


def light_state_monitoring():
    """指示灯状态监控示例"""
    print("\n" + "=" * 50)
    print("指示灯状态变化监控示例（模拟）")
    print("=" * 50)

    from retrosight.recognition.light import LightRecognizer, LightConfig, LightColor

    recognizer = LightRecognizer(LightConfig(
        expected_colors=[LightColor.GREEN, LightColor.RED]
    ))

    # 模拟状态变化序列
    state_sequence = [
        ("green", True),   # 启动
        ("green", True),   # 运行中
        ("green", True),   # 运行中
        ("yellow", True),  # 警告
        ("yellow", True),  # 警告
        ("red", True),     # 故障
        ("red", True),     # 故障
        ("green", True),   # 恢复
    ]

    print("\n状态变化监控:")
    print("-" * 50)

    previous_state = None
    for i, (color, is_on) in enumerate(state_sequence):
        image = create_light_image(color, is_on)
        result = recognizer.detect(image)

        current_state = str(result.color)

        # 检测状态变化
        if previous_state is not None and current_state != previous_state:
            print(f"[{i:2d}] 状态变化: {previous_state} -> {current_state}")
        else:
            print(f"[{i:2d}] 当前状态: {current_state}")

        previous_state = current_state

    print("-" * 50)


def brightness_threshold_example():
    """亮度阈值示例"""
    print("\n" + "=" * 50)
    print("亮度检测示例")
    print("=" * 50)

    # 创建不同亮度的灯
    print("\n测试不同亮度级别:")

    for brightness in [255, 200, 150, 100, 50]:
        # 创建自定义亮度的图像
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (50, 50, 50)

        # 绘制指定亮度的绿灯
        green_color = (0, brightness, 0)
        cv2.circle(img, (50, 50), 30, green_color, -1)

        # 计算区域平均亮度
        roi = img[20:80, 20:80]
        avg_brightness = np.mean(roi)

        # 判断状态
        threshold = 80
        status = "ON" if avg_brightness > threshold else "OFF"

        print(f"   亮度值 {brightness:3d} -> 平均亮度 {avg_brightness:.1f} -> 状态: {status}")


if __name__ == "__main__":
    # 基础检测
    basic_light_detection()

    # Andon 灯检测
    andon_detection_example()

    # 状态监控
    light_state_monitoring()

    # 亮度阈值
    brightness_threshold_example()

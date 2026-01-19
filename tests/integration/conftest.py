"""
集成测试共享 Fixtures

提供测试图像、模拟数据等共享资源
"""

import pytest
import numpy as np
import cv2
from typing import Generator
import tempfile
import os


@pytest.fixture
def sample_digital_image() -> np.ndarray:
    """生成模拟七段数码管图像"""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # 深灰背景

    # 绘制简单的数字 "123"
    cv2.putText(img, "123", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    return img


@pytest.fixture
def sample_gauge_image() -> np.ndarray:
    """生成模拟指针仪表图像"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)  # 白色背景

    center = (150, 150)
    radius = 120

    # 绘制表盘
    cv2.circle(img, center, radius, (0, 0, 0), 2)
    cv2.circle(img, center, 5, (0, 0, 0), -1)

    # 绘制指针（指向约 45 度位置）
    angle_rad = np.radians(45)
    tip_x = int(center[0] + radius * 0.8 * np.cos(angle_rad))
    tip_y = int(center[1] - radius * 0.8 * np.sin(angle_rad))
    cv2.line(img, center, (tip_x, tip_y), (255, 0, 0), 3)

    return img


@pytest.fixture
def sample_light_image_green() -> np.ndarray:
    """生成模拟绿色指示灯图像"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    # 绘制亮起的绿灯
    cv2.circle(img, (50, 50), 30, (0, 255, 0), -1)

    return img


@pytest.fixture
def sample_light_image_red() -> np.ndarray:
    """生成模拟红色指示灯图像"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    # 绘制亮起的红灯
    cv2.circle(img, (50, 50), 30, (0, 0, 255), -1)

    return img


@pytest.fixture
def sample_switch_on_image() -> np.ndarray:
    """生成模拟开关（ON 状态）图像"""
    img = np.zeros((100, 60, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)

    # 开关底座
    cv2.rectangle(img, (10, 30), (50, 70), (100, 100, 100), -1)

    # 开关拨杆（向上 = ON）
    cv2.rectangle(img, (20, 10), (40, 40), (50, 50, 50), -1)

    return img


@pytest.fixture
def sample_switch_off_image() -> np.ndarray:
    """生成模拟开关（OFF 状态）图像"""
    img = np.zeros((100, 60, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)

    # 开关底座
    cv2.rectangle(img, (10, 30), (50, 70), (100, 100, 100), -1)

    # 开关拨杆（向下 = OFF）
    cv2.rectangle(img, (20, 60), (40, 90), (50, 50, 50), -1)

    return img


@pytest.fixture
def temp_config_file() -> Generator[str, None, None]:
    """创建临时配置文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name

    yield path

    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_calibration_file() -> Generator[str, None, None]:
    """创建临时校准文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name

    yield path

    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_mqtt_config() -> dict:
    """模拟 MQTT 配置"""
    return {
        "broker": "localhost",
        "port": 1883,
        "topic_prefix": "test/retrosight",
        "client_id": "test_client"
    }


@pytest.fixture
def mock_modbus_config() -> dict:
    """模拟 Modbus 配置"""
    return {
        "host": "0.0.0.0",
        "port": 5020,  # 使用非标准端口避免冲突
        "unit_id": 1
    }

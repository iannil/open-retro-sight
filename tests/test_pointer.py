"""
指针识别模块单元测试
"""

import pytest
import numpy as np
import math
from unittest.mock import patch, MagicMock

from retrosight.recognition.pointer import (
    PointerRecognizer,
    PointerResult,
    GaugeConfig,
    GaugeType,
    recognize_gauge,
)


class TestGaugeType:
    """仪表类型测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert GaugeType.CIRCULAR.value == "circular"
        assert GaugeType.SEMICIRCLE.value == "semicircle"
        assert GaugeType.ARC.value == "arc"
        assert GaugeType.LINEAR.value == "linear"


class TestGaugeConfig:
    """仪表配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = GaugeConfig()
        assert config.gauge_type == GaugeType.CIRCULAR
        assert config.min_angle == 225.0
        assert config.max_angle == -45.0
        assert config.min_value == 0.0
        assert config.max_value == 100.0

    def test_custom_values(self):
        """测试自定义值"""
        config = GaugeConfig(
            center=(100, 100),
            radius=80,
            min_value=0,
            max_value=200,
            unit="MPa"
        )
        assert config.center == (100, 100)
        assert config.radius == 80
        assert config.unit == "MPa"


class TestPointerResult:
    """指针结果测试"""

    def test_creation(self):
        """测试创建"""
        result = PointerResult(
            angle=135.0,
            value=50.0,
            confidence=0.9
        )
        assert result.angle == 135.0
        assert result.value == 50.0
        assert result.confidence == 0.9

    def test_default_values(self):
        """测试默认值"""
        result = PointerResult(angle=0, value=0)
        assert result.confidence == 0.0
        assert result.center is None
        assert result.tip is None
        assert result.raw_lines == []


class TestPointerRecognizer:
    """指针识别器测试"""

    def test_initialization(self):
        """测试初始化"""
        recognizer = PointerRecognizer()
        assert recognizer.config.gauge_type == GaugeType.CIRCULAR

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = GaugeConfig(min_value=0, max_value=200)
        recognizer = PointerRecognizer(config)
        assert recognizer.config.max_value == 200

    def test_angle_to_value_linear(self):
        """测试角度到数值的线性映射"""
        config = GaugeConfig(
            min_angle=0,
            max_angle=180,
            min_value=0,
            max_value=100
        )
        recognizer = PointerRecognizer(config)

        # 中间角度应该对应中间值
        value = recognizer._angle_to_value(90)
        assert 45 <= value <= 55  # 允许一定误差

    def test_point_line_distance(self):
        """测试点到线段距离"""
        recognizer = PointerRecognizer()

        # 点在线段上
        dist = recognizer._point_line_distance((5, 0), (0, 0), (10, 0))
        assert dist == 0

        # 点垂直于线段
        dist = recognizer._point_line_distance((5, 10), (0, 0), (10, 0))
        assert dist == 10

    def test_preprocess_grayscale(self):
        """测试灰度图预处理"""
        recognizer = PointerRecognizer()
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        result = recognizer._preprocess(gray_image)
        assert result is not None

    def test_preprocess_color(self):
        """测试彩色图预处理"""
        recognizer = PointerRecognizer()
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognizer._preprocess(color_image)
        assert result is not None

    def test_preprocess_red_pointer(self):
        """测试红色指针预处理"""
        config = GaugeConfig(pointer_color="red")
        recognizer = PointerRecognizer(config)
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognizer._preprocess(color_image)
        assert result is not None

    def test_detect_dial(self):
        """测试表盘检测"""
        recognizer = PointerRecognizer()
        # 创建带圆形的测试图像
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        center, radius = recognizer._detect_dial(image)

        # 应该返回默认中心
        assert center is not None
        assert radius is not None

    def test_recognize_empty_image(self):
        """测试空图像识别"""
        config = GaugeConfig(center=(100, 100), radius=80)
        recognizer = PointerRecognizer(config)
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        result = recognizer.recognize(image)

        assert isinstance(result, PointerResult)

    def test_visualize(self):
        """测试可视化"""
        config = GaugeConfig(center=(100, 100), radius=80)
        recognizer = PointerRecognizer(config)
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        result = PointerResult(
            angle=45,
            value=25,
            confidence=0.8,
            tip=(150, 50)
        )

        output = recognizer.visualize(image, result)
        assert output.shape == image.shape


class TestRecognizeGauge:
    """便捷函数测试"""

    def test_recognize_gauge(self):
        """测试便捷识别函数"""
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        result = recognize_gauge(
            image,
            min_value=0,
            max_value=100,
            unit="bar"
        )

        assert isinstance(result, PointerResult)


class TestPointerMath:
    """指针数学计算测试"""

    def test_calculate_angle_from_lines(self):
        """测试从线段计算角度"""
        config = GaugeConfig(center=(100, 100), radius=80)
        recognizer = PointerRecognizer(config)

        # 创建一条从中心向上的线段
        lines = [np.array([100, 100, 100, 20])]

        angle, tip = recognizer._calculate_angle_from_lines(lines)

        # 向上应该是0度（12点钟方向）
        assert 355 <= angle or angle <= 5

    def test_group_lines_by_length(self):
        """测试按长度分组线段"""
        recognizer = PointerRecognizer()

        lines = [
            [np.array([0, 0, 100, 0])],   # 长度 100
            [np.array([0, 0, 50, 0])],    # 长度 50
            [np.array([0, 0, 25, 0])],    # 长度 25
        ]

        groups = recognizer._group_lines_by_length(lines, 2)

        assert len(groups) <= 2

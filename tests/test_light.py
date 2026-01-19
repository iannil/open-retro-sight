"""
指示灯识别模块单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from retrosight.recognition.light import (
    LightColor,
    LightState,
    ColorRange,
    LightConfig,
    LightResult,
    LightRecognizer,
    AndonMonitor,
    detect_light,
    detect_andon,
    DEFAULT_COLOR_RANGES,
)


class TestLightColor:
    """指示灯颜色枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert LightColor.RED.value == "red"
        assert LightColor.YELLOW.value == "yellow"
        assert LightColor.GREEN.value == "green"
        assert LightColor.BLUE.value == "blue"
        assert LightColor.UNKNOWN.value == "unknown"


class TestLightState:
    """指示灯状态枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert LightState.OFF.value == "off"
        assert LightState.ON.value == "on"
        assert LightState.BLINKING.value == "blinking"


class TestColorRange:
    """颜色范围测试"""

    def test_creation(self):
        """测试创建"""
        cr = ColorRange((0, 100, 100), (10, 255, 255))
        assert cr.lower == (0, 100, 100)
        assert cr.upper == (10, 255, 255)


class TestLightConfig:
    """指示灯配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = LightConfig()
        assert config.region is None
        assert config.brightness_threshold == 100
        assert config.min_area == 50
        assert config.max_area == 10000
        assert config.blink_window == 30
        assert config.blink_threshold == 2

    def test_custom_values(self):
        """测试自定义值"""
        config = LightConfig(
            region=(10, 20, 50, 50),
            brightness_threshold=150,
            min_area=100
        )
        assert config.region == (10, 20, 50, 50)
        assert config.brightness_threshold == 150
        assert config.min_area == 100


class TestLightResult:
    """指示灯结果测试"""

    def test_creation(self):
        """测试创建"""
        result = LightResult(
            color=LightColor.RED,
            state=LightState.ON,
            brightness=200.0,
            confidence=0.95
        )
        assert result.color == LightColor.RED
        assert result.state == LightState.ON
        assert result.brightness == 200.0
        assert result.confidence == 0.95

    def test_default_values(self):
        """测试默认值"""
        result = LightResult(
            color=LightColor.GREEN,
            state=LightState.OFF
        )
        assert result.brightness == 0.0
        assert result.confidence == 0.0
        assert result.position is None
        assert result.area == 0
        assert result.blink_frequency == 0.0


class TestDefaultColorRanges:
    """默认颜色范围测试"""

    def test_all_colors_defined(self):
        """测试所有颜色都有定义"""
        assert LightColor.RED in DEFAULT_COLOR_RANGES
        assert LightColor.YELLOW in DEFAULT_COLOR_RANGES
        assert LightColor.GREEN in DEFAULT_COLOR_RANGES
        assert LightColor.BLUE in DEFAULT_COLOR_RANGES
        assert LightColor.WHITE in DEFAULT_COLOR_RANGES

    def test_red_has_two_ranges(self):
        """测试红色有两个范围（环绕色相）"""
        assert len(DEFAULT_COLOR_RANGES[LightColor.RED]) == 2


class TestLightRecognizer:
    """指示灯识别器测试"""

    @pytest.fixture
    def recognizer(self):
        """创建识别器"""
        return LightRecognizer()

    @pytest.fixture
    def green_image(self):
        """创建绿色测试图像"""
        # 创建 100x100 的绿色图像
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 在中心绘制绿色圆
        cv2 = pytest.importorskip("cv2")
        cv2.circle(image, (50, 50), 20, (0, 255, 0), -1)
        return image

    @pytest.fixture
    def red_image(self):
        """创建红色测试图像"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2 = pytest.importorskip("cv2")
        cv2.circle(image, (50, 50), 20, (0, 0, 255), -1)
        return image

    @pytest.fixture
    def black_image(self):
        """创建黑色测试图像（灯熄灭）"""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_initialization(self, recognizer):
        """测试初始化"""
        assert recognizer.config is not None
        assert len(recognizer._history) == 0

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = LightConfig(brightness_threshold=150)
        recognizer = LightRecognizer(config)
        assert recognizer.config.brightness_threshold == 150

    def test_detect_green_light(self, recognizer, green_image):
        """测试检测绿灯"""
        result = recognizer.detect(green_image)
        # 应该检测到绿色或亮的状态
        assert result is not None
        assert isinstance(result, LightResult)

    def test_detect_off_light(self, recognizer, black_image):
        """测试检测熄灭的灯"""
        result = recognizer.detect(black_image)
        assert result.state == LightState.OFF or result.color == LightColor.UNKNOWN

    def test_extract_roi(self, recognizer):
        """测试 ROI 提取"""
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        # 无 ROI
        roi = recognizer._extract_roi(image)
        assert roi.shape == (200, 200, 3)

        # 有 ROI
        recognizer.config.region = (50, 50, 100, 100)
        roi = recognizer._extract_roi(image)
        assert roi.shape == (100, 100, 3)

    def test_reset(self, recognizer, green_image):
        """测试重置"""
        # 先检测几次
        for _ in range(5):
            recognizer.detect(green_image)

        assert len(recognizer._history) > 0

        recognizer.reset()
        assert len(recognizer._history) == 0
        assert recognizer._last_state is None
        assert recognizer._state_changes == 0

    def test_detect_multiple_empty(self, recognizer, black_image):
        """测试多灯检测（空图像）"""
        results = recognizer.detect_multiple(black_image)
        assert isinstance(results, list)

    def test_detect_andon(self, recognizer, black_image):
        """测试 Andon 灯检测"""
        andon = recognizer.detect_andon(black_image)
        assert "red" in andon
        assert "yellow" in andon
        assert "green" in andon

    def test_visualize(self, recognizer, green_image):
        """测试可视化"""
        results = [
            LightResult(
                color=LightColor.GREEN,
                state=LightState.ON,
                position=(50, 50),
                area=100
            )
        ]
        output = recognizer.visualize(green_image, results)
        assert output.shape == green_image.shape


class TestAndonMonitor:
    """Andon 监控器测试"""

    @pytest.fixture
    def monitor(self):
        """创建监控器"""
        return AndonMonitor()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        return np.zeros((200, 200, 3), dtype=np.uint8)

    def test_initialization(self, monitor):
        """测试初始化"""
        assert monitor._running_time == 0.0
        assert monitor._idle_time == 0.0
        assert monitor._fault_time == 0.0
        assert monitor._current_state == "unknown"

    def test_update(self, monitor, test_image):
        """测试更新"""
        status = monitor.update(test_image)

        assert "state" in status
        assert "andon" in status
        assert "runtime" in status
        assert "idle_time" in status
        assert "fault_time" in status
        assert "total_time" in status
        assert "availability" in status

    def test_reset(self, monitor, test_image):
        """测试重置"""
        # 先更新几次
        for _ in range(3):
            monitor.update(test_image)

        monitor.reset()

        assert monitor._running_time == 0.0
        assert monitor._idle_time == 0.0
        assert monitor._fault_time == 0.0


class TestConvenienceFunctions:
    """便捷函数测试"""

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_detect_light(self, test_image):
        """测试 detect_light"""
        result = detect_light(test_image)
        assert isinstance(result, LightResult)

    def test_detect_light_with_region(self, test_image):
        """测试带区域的 detect_light"""
        result = detect_light(test_image, region=(10, 10, 50, 50))
        assert isinstance(result, LightResult)

    def test_detect_andon_func(self, test_image):
        """测试 detect_andon"""
        result = detect_andon(test_image)
        assert isinstance(result, dict)
        assert "red" in result
        assert "yellow" in result
        assert "green" in result

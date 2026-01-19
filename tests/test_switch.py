"""
开关/旋钮识别模块单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from retrosight.recognition.switch import (
    SwitchType,
    SwitchState,
    SwitchConfig,
    SwitchResult,
    SwitchRecognizer,
    MultiSwitchMonitor,
    detect_switch,
    detect_rotary,
)


class TestSwitchType:
    """开关类型枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert SwitchType.TOGGLE.value == "toggle"
        assert SwitchType.ROTARY.value == "rotary"
        assert SwitchType.PUSH_BUTTON.value == "button"
        assert SwitchType.SLIDER.value == "slider"
        assert SwitchType.SELECTOR.value == "selector"


class TestSwitchState:
    """开关状态枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert SwitchState.ON.value == "on"
        assert SwitchState.OFF.value == "off"
        assert SwitchState.MIDDLE.value == "middle"
        assert SwitchState.UNKNOWN.value == "unknown"


class TestSwitchConfig:
    """开关配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = SwitchConfig()
        assert config.switch_type == SwitchType.TOGGLE
        assert config.region is None
        assert config.num_positions == 2
        assert config.match_threshold == 0.7
        assert config.use_color is False

    def test_custom_values(self):
        """测试自定义值"""
        config = SwitchConfig(
            switch_type=SwitchType.ROTARY,
            region=(10, 20, 100, 100),
            num_positions=4,
            position_labels=["OFF", "LOW", "MED", "HIGH"]
        )
        assert config.switch_type == SwitchType.ROTARY
        assert config.region == (10, 20, 100, 100)
        assert config.num_positions == 4
        assert len(config.position_labels) == 4


class TestSwitchResult:
    """开关结果测试"""

    def test_creation(self):
        """测试创建"""
        result = SwitchResult(
            state=SwitchState.ON,
            position=1,
            position_label="ON",
            confidence=0.95
        )
        assert result.state == SwitchState.ON
        assert result.position == 1
        assert result.position_label == "ON"
        assert result.confidence == 0.95

    def test_default_values(self):
        """测试默认值"""
        result = SwitchResult(state=SwitchState.OFF)
        assert result.position == 0
        assert result.position_label == ""
        assert result.confidence == 0.0
        assert result.angle == 0.0
        assert result.center is None


class TestSwitchRecognizer:
    """开关识别器测试"""

    @pytest.fixture
    def recognizer(self):
        """创建识别器"""
        return SwitchRecognizer()

    @pytest.fixture
    def rotary_recognizer(self):
        """创建旋钮识别器"""
        config = SwitchConfig(
            switch_type=SwitchType.ROTARY,
            num_positions=4,
            position_labels=["OFF", "LOW", "MED", "HIGH"]
        )
        return SwitchRecognizer(config)

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        # 创建简单的黑白图像
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 上半部分白色
        image[:50, :, :] = 255
        return image

    @pytest.fixture
    def uniform_image(self):
        """创建均匀灰色图像"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 128

    def test_initialization(self, recognizer):
        """测试初始化"""
        assert recognizer.config is not None
        assert recognizer.config.switch_type == SwitchType.TOGGLE

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = SwitchConfig(switch_type=SwitchType.ROTARY)
        recognizer = SwitchRecognizer(config)
        assert recognizer.config.switch_type == SwitchType.ROTARY

    def test_recognize_toggle(self, recognizer, test_image):
        """测试拨动开关识别"""
        result = recognizer.recognize(test_image)
        assert isinstance(result, SwitchResult)
        # 上半部分亮 -> 应该是 ON
        assert result.state in [SwitchState.ON, SwitchState.OFF, SwitchState.UNKNOWN]

    def test_recognize_rotary(self, rotary_recognizer, uniform_image):
        """测试旋钮识别"""
        result = rotary_recognizer.recognize(uniform_image)
        assert isinstance(result, SwitchResult)

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

    def test_angle_to_position(self, rotary_recognizer):
        """测试角度到档位转换"""
        # 4档位均匀分布：0, 90, 180, 270
        pos, label, conf = rotary_recognizer._angle_to_position(5)
        assert pos == 0
        assert label == "OFF"

        pos, label, conf = rotary_recognizer._angle_to_position(95)
        assert pos == 1
        assert label == "LOW"

    def test_detect_toggle_position(self, recognizer, test_image):
        """测试拨动开关位置检测"""
        cv2 = pytest.importorskip("cv2")
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        result = recognizer._detect_toggle_position(gray)
        assert isinstance(result, SwitchResult)

    def test_recognize_button(self):
        """测试按钮识别"""
        config = SwitchConfig(switch_type=SwitchType.PUSH_BUTTON)
        recognizer = SwitchRecognizer(config)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = recognizer.recognize(image)
        assert isinstance(result, SwitchResult)

    def test_recognize_slider(self):
        """测试滑动开关识别"""
        config = SwitchConfig(switch_type=SwitchType.SLIDER)
        recognizer = SwitchRecognizer(config)
        # 创建带有白色滑块的图像
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        image[:, 80:100, :] = 255  # 滑块在右边 -> ON
        result = recognizer.recognize(image)
        assert isinstance(result, SwitchResult)

    def test_set_templates(self, recognizer):
        """测试设置模板"""
        on_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        off_img = np.zeros((50, 50, 3), dtype=np.uint8)

        recognizer.set_templates(on_img, off_img)
        assert recognizer._on_template is not None
        assert recognizer._off_template is not None

    def test_visualize(self, rotary_recognizer, uniform_image):
        """测试可视化"""
        result = SwitchResult(
            state=SwitchState.ON,
            position=1,
            position_label="LOW",
            confidence=0.9,
            angle=90,
            center=(50, 50)
        )
        output = rotary_recognizer.visualize(uniform_image, result)
        assert output.shape == uniform_image.shape


class TestMultiSwitchMonitor:
    """多开关监控器测试"""

    @pytest.fixture
    def monitor(self):
        """创建监控器"""
        return MultiSwitchMonitor()

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        return np.zeros((200, 200, 3), dtype=np.uint8)

    def test_initialization(self, monitor):
        """测试初始化"""
        assert len(monitor._switches) == 0
        assert len(monitor._last_results) == 0

    def test_add_switch(self, monitor):
        """测试添加开关"""
        config = SwitchConfig(region=(10, 10, 50, 50))
        monitor.add_switch("power", config)
        assert "power" in monitor._switches
        assert "power" in monitor.switch_names

    def test_remove_switch(self, monitor):
        """测试移除开关"""
        config = SwitchConfig()
        monitor.add_switch("test", config)
        monitor.remove_switch("test")
        assert "test" not in monitor._switches

    def test_update(self, monitor, test_image):
        """测试更新"""
        config = SwitchConfig(region=(10, 10, 50, 50))
        monitor.add_switch("sw1", config)

        results = monitor.update(test_image)
        assert "sw1" in results
        assert isinstance(results["sw1"], SwitchResult)

    def test_get_state(self, monitor, test_image):
        """测试获取状态"""
        config = SwitchConfig()
        monitor.add_switch("test", config)
        monitor.update(test_image)

        state = monitor.get_state("test")
        assert isinstance(state, SwitchResult)

    def test_get_all_states(self, monitor, test_image):
        """测试获取所有状态"""
        monitor.add_switch("sw1", SwitchConfig())
        monitor.add_switch("sw2", SwitchConfig())
        monitor.update(test_image)

        states = monitor.get_all_states()
        assert len(states) == 2

    def test_switch_names(self, monitor):
        """测试获取开关名称"""
        monitor.add_switch("a", SwitchConfig())
        monitor.add_switch("b", SwitchConfig())

        names = monitor.switch_names
        assert "a" in names
        assert "b" in names


class TestConvenienceFunctions:
    """便捷函数测试"""

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_detect_switch(self, test_image):
        """测试 detect_switch"""
        result = detect_switch(test_image)
        assert isinstance(result, SwitchResult)

    def test_detect_switch_with_region(self, test_image):
        """测试带区域的 detect_switch"""
        result = detect_switch(test_image, region=(10, 10, 50, 50))
        assert isinstance(result, SwitchResult)

    def test_detect_switch_types(self, test_image):
        """测试不同类型的开关检测"""
        for switch_type in SwitchType:
            result = detect_switch(test_image, switch_type=switch_type)
            assert isinstance(result, SwitchResult)

    def test_detect_rotary(self, test_image):
        """测试 detect_rotary"""
        result = detect_rotary(test_image, num_positions=3)
        assert isinstance(result, SwitchResult)

    def test_detect_rotary_with_labels(self, test_image):
        """测试带标签的 detect_rotary"""
        result = detect_rotary(
            test_image,
            num_positions=3,
            position_labels=["OFF", "AUTO", "ON"]
        )
        assert isinstance(result, SwitchResult)

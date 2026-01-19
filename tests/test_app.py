"""
Web 界面模块测试

测试内容:
- AppConfig 配置类
- 配置文件保存/加载
- MQTT 连接测试
- 滤波器创建
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from retrosight.ui.app import (
    AppConfig,
    save_config,
    load_config,
    test_mqtt_connection,
    create_filter,
    create_placeholder_image,
)


class TestAppConfig:
    """AppConfig 配置类测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = AppConfig()

        # 摄像头默认值
        assert config.camera_source == 0
        assert config.camera_width == 640
        assert config.camera_height == 480
        assert config.camera_fps == 30

        # ROI 默认值
        assert config.roi_enabled is False
        assert config.roi_x == 0
        assert config.roi_y == 0

        # MQTT 默认值
        assert config.mqtt_enabled is False
        assert config.mqtt_host == "localhost"
        assert config.mqtt_port == 1883

        # 滤波默认值
        assert config.filter_enabled is True
        assert config.filter_type == "kalman"

    def test_custom_config(self):
        """测试自定义配置"""
        config = AppConfig(
            camera_source=1,
            camera_width=1280,
            mqtt_enabled=True,
            mqtt_host="broker.example.com"
        )

        assert config.camera_source == 1
        assert config.camera_width == 1280
        assert config.mqtt_enabled is True
        assert config.mqtt_host == "broker.example.com"


class TestConfigPersistence:
    """配置持久化测试"""

    def test_save_and_load_config(self):
        """测试保存和加载配置"""
        config = AppConfig(
            camera_source=2,
            camera_width=800,
            mqtt_enabled=True,
            mqtt_host="test.broker.com",
            mqtt_port=8883
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            # 保存
            result = save_config(config, path)
            assert result is True
            assert os.path.exists(path)

            # 验证文件内容
            with open(path, 'r') as f:
                data = json.load(f)
                assert data['camera_source'] == 2
                assert data['mqtt_host'] == "test.broker.com"

            # 加载
            loaded = load_config(path)
            assert loaded is not None
            assert loaded.camera_source == 2
            assert loaded.camera_width == 800
            assert loaded.mqtt_enabled is True
            assert loaded.mqtt_host == "test.broker.com"
            assert loaded.mqtt_port == 8883

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_nonexistent_config(self):
        """测试加载不存在的配置文件"""
        result = load_config("/nonexistent/path/config.json")
        assert result is None

    def test_save_config_error_handling(self):
        """测试保存配置错误处理"""
        config = AppConfig()
        # 尝试保存到无效路径
        result = save_config(config, "/invalid/path/config.json")
        assert result is False


class TestMQTTConnectionTest:
    """MQTT 连接测试"""

    @patch('retrosight.ui.app.mqtt')
    def test_successful_connection(self, mock_mqtt_module):
        """测试成功连接"""
        # 模拟 MQTT 客户端
        mock_client = MagicMock()
        mock_mqtt_module.Client.return_value = mock_client
        mock_mqtt_module.MQTTv311 = 4

        # 模拟连接成功回调
        def connect_side_effect(host, port, keepalive):
            # 触发 on_connect 回调
            mock_client.on_connect(mock_client, None, None, 0)

        mock_client.connect.side_effect = connect_side_effect

        config = AppConfig(
            mqtt_host="localhost",
            mqtt_port=1883
        )

        success, message = test_mqtt_connection(config)

        assert success is True
        assert "成功" in message

    @patch('retrosight.ui.app.mqtt')
    def test_connection_auth_error(self, mock_mqtt_module):
        """测试认证错误"""
        mock_client = MagicMock()
        mock_mqtt_module.Client.return_value = mock_client
        mock_mqtt_module.MQTTv311 = 4

        def connect_side_effect(host, port, keepalive):
            mock_client.on_connect(mock_client, None, None, 4)  # 认证失败

        mock_client.connect.side_effect = connect_side_effect

        config = AppConfig(
            mqtt_host="localhost",
            mqtt_username="user",
            mqtt_password="wrong"
        )

        success, message = test_mqtt_connection(config)

        assert success is False
        assert "密码" in message or "错误" in message

    def test_connection_refused(self):
        """测试连接被拒绝"""
        config = AppConfig(
            mqtt_host="localhost",
            mqtt_port=9999  # 不存在的端口
        )

        success, message = test_mqtt_connection(config)

        assert success is False
        assert "连接" in message or "失败" in message or "拒绝" in message

    def test_connection_with_auth(self):
        """测试带认证的连接"""
        config = AppConfig(
            mqtt_host="localhost",
            mqtt_port=1883,
            mqtt_username="testuser",
            mqtt_password="testpass"
        )

        # 不实际连接，只测试函数不会抛出异常
        success, message = test_mqtt_connection(config)
        assert isinstance(success, bool)
        assert isinstance(message, str)


class TestFilterCreation:
    """滤波器创建测试"""

    def test_create_kalman_filter(self):
        """测试创建卡尔曼滤波器"""
        config = AppConfig(
            filter_enabled=True,
            filter_type="kalman"
        )

        filter_obj = create_filter(config)

        assert filter_obj is not None
        # 测试滤波器可以工作
        result = filter_obj.update(10.0)
        assert isinstance(result, float)

    def test_create_moving_average_filter(self):
        """测试创建滑动平均滤波器"""
        config = AppConfig(
            filter_enabled=True,
            filter_type="moving_average",
            filter_window_size=5
        )

        filter_obj = create_filter(config)

        assert filter_obj is not None
        # 测试滤波器可以工作
        for i in range(10):
            result = filter_obj.update(float(i))
        assert isinstance(result, float)

    def test_create_exponential_filter(self):
        """测试创建指数平滑滤波器"""
        config = AppConfig(
            filter_enabled=True,
            filter_type="exponential",
            filter_alpha=0.3
        )

        filter_obj = create_filter(config)

        assert filter_obj is not None
        result = filter_obj.update(10.0)
        assert isinstance(result, float)

    def test_filter_disabled(self):
        """测试禁用滤波器"""
        config = AppConfig(filter_enabled=False)

        filter_obj = create_filter(config)

        assert filter_obj is None


class TestPlaceholderImage:
    """占位图像测试"""

    def test_create_placeholder_image(self):
        """测试创建占位图像"""
        img = create_placeholder_image()

        assert img is not None
        assert isinstance(img, np.ndarray)
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8


class TestCalibrationIntegration:
    """指针校准集成测试"""

    def test_calibration_data_persistence(self):
        """测试校准数据持久化"""
        from retrosight.recognition.pointer import (
            PointerRecognizer,
            GaugeConfig,
            CalibrationData,
            CalibrationPoint
        )

        # 创建识别器并进行校准
        config = GaugeConfig(
            min_value=0,
            max_value=100,
            unit="MPa"
        )
        recognizer = PointerRecognizer(config)

        # 两点校准
        recognizer.calibrate_two_point(
            angle1=45.0, value1=0.0,
            angle2=315.0, value2=100.0
        )

        assert recognizer.calibration is not None
        assert recognizer.calibration.is_valid()
        assert len(recognizer.calibration.points) == 2

        # 保存校准数据
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            recognizer.save_calibration(path)
            assert os.path.exists(path)

            # 创建新识别器并加载校准数据
            new_recognizer = PointerRecognizer(GaugeConfig())
            new_recognizer.load_calibration(path)

            assert new_recognizer.calibration is not None
            assert new_recognizer.calibration.is_valid()
            assert len(new_recognizer.calibration.points) == 2

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_three_point_calibration(self):
        """测试三点校准"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

        recognizer = PointerRecognizer(GaugeConfig())

        # 三点校准（非线性）
        recognizer.calibrate_three_point(
            angle1=45.0, value1=0.0,
            angle2=180.0, value2=40.0,  # 非线性中间点
            angle3=315.0, value3=100.0
        )

        assert recognizer.calibration is not None
        assert recognizer.calibration.method == "polynomial"
        assert len(recognizer.calibration.coefficients) == 3  # 二次多项式有3个系数

    def test_clear_calibration(self):
        """测试清除校准"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

        recognizer = PointerRecognizer(GaugeConfig())
        recognizer.calibrate_two_point(45.0, 0.0, 315.0, 100.0)

        assert recognizer.calibration is not None

        recognizer.clear_calibration()

        assert recognizer.calibration is None

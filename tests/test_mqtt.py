"""
MQTT 模块单元测试
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from retrosight.output.mqtt import (
    MQTTConfig,
    MQTTPublisher,
    MQTTSubscriber,
    SensorData,
    create_publisher,
)


class TestMQTTConfig:
    """MQTT 配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = MQTTConfig()
        assert config.host == "localhost"
        assert config.port == 1883
        assert config.qos == 1
        assert config.topic_prefix == "retrosight"
        assert config.max_buffer_size == 1000

    def test_custom_values(self):
        """测试自定义值"""
        config = MQTTConfig(
            host="mqtt.example.com",
            port=8883,
            username="user",
            password="pass",
            use_tls=True
        )
        assert config.host == "mqtt.example.com"
        assert config.port == 8883
        assert config.username == "user"
        assert config.use_tls is True


class TestSensorData:
    """传感器数据测试"""

    def test_creation(self):
        """测试创建"""
        data = SensorData(
            sensor_id="temp_01",
            value=25.5,
            unit="°C",
            confidence=0.95
        )
        assert data.sensor_id == "temp_01"
        assert data.value == 25.5
        assert data.unit == "°C"
        assert data.confidence == 0.95

    def test_auto_timestamp(self):
        """测试自动时间戳"""
        data = SensorData(sensor_id="test", value=1.0)
        assert data.timestamp is not None
        assert data.timestamp.endswith("Z")

    def test_custom_timestamp(self):
        """测试自定义时间戳"""
        custom_ts = "2024-01-01T00:00:00Z"
        data = SensorData(
            sensor_id="test",
            value=1.0,
            timestamp=custom_ts
        )
        assert data.timestamp == custom_ts

    def test_to_json(self):
        """测试 JSON 序列化"""
        data = SensorData(
            sensor_id="test",
            value=1.5,
            unit="unit",
            timestamp="2024-01-01T00:00:00Z"
        )
        json_str = data.to_json()
        parsed = json.loads(json_str)

        assert parsed["sensor_id"] == "test"
        assert parsed["value"] == 1.5
        assert parsed["unit"] == "unit"

    def test_metadata(self):
        """测试元数据"""
        data = SensorData(
            sensor_id="test",
            value=1.0,
            metadata={"location": "factory1", "device": "CNC01"}
        )
        assert data.metadata["location"] == "factory1"
        assert data.metadata["device"] == "CNC01"


class TestMQTTPublisher:
    """MQTT 发布器测试"""

    def test_initialization(self):
        """测试初始化"""
        publisher = MQTTPublisher()
        assert publisher._connected is False
        assert publisher._running is False

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = MQTTConfig(host="custom.host", port=1884)
        publisher = MQTTPublisher(config)
        assert publisher.config.host == "custom.host"
        assert publisher.config.port == 1884

    @patch("retrosight.output.mqtt.MQTTPublisher._init_paho", return_value=None)
    def test_publish_when_disconnected(self, mock_init):
        """测试断开时发布（缓存）"""
        publisher = MQTTPublisher()
        publisher._running = True
        publisher._connected = False

        data = SensorData(sensor_id="test", value=1.0)
        result = publisher.publish(data)

        assert result is True
        assert publisher.buffer_size == 1

    def test_publish_value_convenience(self):
        """测试便捷发布方法"""
        publisher = MQTTPublisher()
        publisher._running = True

        result = publisher.publish_value(
            sensor_id="pressure_01",
            value=1.23,
            unit="MPa",
            confidence=0.98
        )

        assert result is True
        assert publisher.buffer_size == 1

    def test_buffer_size_limit(self):
        """测试缓冲区大小限制"""
        config = MQTTConfig(max_buffer_size=5)
        publisher = MQTTPublisher(config)
        publisher._running = True

        # 发送超过缓冲区大小的消息
        for i in range(10):
            publisher.publish_value(f"sensor_{i}", float(i))

        # 缓冲区应该不超过限制
        assert publisher.buffer_size <= 5

    def test_is_connected_property(self):
        """测试 is_connected 属性"""
        publisher = MQTTPublisher()
        assert publisher.is_connected is False

        publisher._connected = True
        assert publisher.is_connected is True

    def test_on_connect_callback(self):
        """测试连接回调注册"""
        publisher = MQTTPublisher()
        callback = Mock()

        publisher.on_connect(callback)

        assert callback in publisher._on_connect_callbacks

    def test_on_disconnect_callback(self):
        """测试断开回调注册"""
        publisher = MQTTPublisher()
        callback = Mock()

        publisher.on_disconnect(callback)

        assert callback in publisher._on_disconnect_callbacks


class TestMQTTSubscriber:
    """MQTT 订阅器测试"""

    def test_initialization(self):
        """测试初始化"""
        subscriber = MQTTSubscriber()
        assert subscriber._running is False
        assert len(subscriber._subscriptions) == 0

    def test_subscribe(self):
        """测试订阅"""
        subscriber = MQTTSubscriber()
        callback = Mock()

        subscriber.subscribe("sensors/#", callback)

        full_topic = f"{subscriber.config.topic_prefix}/sensors/#"
        assert full_topic in subscriber._subscriptions

    def test_topic_matching_exact(self):
        """测试精确主题匹配"""
        subscriber = MQTTSubscriber()
        assert subscriber._topic_matches("a/b/c", "a/b/c") is True
        assert subscriber._topic_matches("a/b/c", "a/b/d") is False

    def test_topic_matching_wildcard_hash(self):
        """测试 # 通配符"""
        subscriber = MQTTSubscriber()
        assert subscriber._topic_matches("a/#", "a/b/c") is True
        assert subscriber._topic_matches("a/#", "a") is False

    def test_topic_matching_wildcard_plus(self):
        """测试 + 通配符"""
        subscriber = MQTTSubscriber()
        assert subscriber._topic_matches("a/+/c", "a/b/c") is True
        assert subscriber._topic_matches("a/+/c", "a/x/c") is True
        assert subscriber._topic_matches("a/+/c", "a/b/d") is False


class TestCreatePublisher:
    """测试便捷函数"""

    def test_create_publisher(self):
        """测试创建发布器"""
        publisher = create_publisher(
            host="mqtt.test.com",
            port=1884,
            topic_prefix="test/prefix"
        )

        assert isinstance(publisher, MQTTPublisher)
        assert publisher.config.host == "mqtt.test.com"
        assert publisher.config.port == 1884
        assert publisher.config.topic_prefix == "test/prefix"

    def test_create_publisher_defaults(self):
        """测试默认值"""
        publisher = create_publisher()

        assert publisher.config.host == "localhost"
        assert publisher.config.port == 1883


class TestMQTTPublisherContextManager:
    """测试上下文管理器"""

    @patch("paho.mqtt.client.Client")
    def test_context_manager(self, mock_client):
        """测试上下文管理器协议"""
        config = MQTTConfig()
        publisher = MQTTPublisher(config)

        # 模拟 __enter__ 和 __exit__
        with patch.object(publisher, 'start', return_value=True):
            with patch.object(publisher, 'stop'):
                with publisher as p:
                    assert p is publisher

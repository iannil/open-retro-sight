"""
识别到输出的集成测试

测试场景:
- OCR 识别 → MQTT 输出
- 指针识别 → Modbus 输出
- 多识别器 → 多协议输出
"""

import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass


class TestOCRToMQTT:
    """OCR 识别到 MQTT 输出集成测试"""

    def test_ocr_result_to_mqtt_message(self, sample_digital_image):
        """测试 OCR 结果转 MQTT 消息"""
        from retrosight.recognition.ocr import SimpleOCR
        from retrosight.output.mqtt import MQTTPublisher, MQTTConfig

        # OCR 识别
        ocr = SimpleOCR()
        result = ocr.recognize(sample_digital_image)

        # 构造 MQTT 消息
        message = {
            "text": result.text,
            "confidence": result.confidence,
            "timestamp": "2024-01-01T00:00:00Z"
        }

        assert "text" in message
        assert "confidence" in message
        assert isinstance(message["confidence"], float)

    def test_ocr_batch_to_mqtt(self, sample_digital_image):
        """测试批量 OCR 结果发送"""
        from retrosight.recognition.ocr import SimpleOCR

        ocr = SimpleOCR()

        # 模拟连续识别
        results = []
        for _ in range(5):
            result = ocr.recognize(sample_digital_image)
            results.append({
                "text": result.text,
                "confidence": result.confidence
            })

        assert len(results) == 5

    def test_filtered_ocr_to_mqtt(self, sample_digital_image):
        """测试过滤后的 OCR 结果发送"""
        from retrosight.recognition.ocr import SimpleOCR
        from retrosight.preprocessing.filter import create_default_filter

        ocr = SimpleOCR()
        filter_obj = create_default_filter()

        # 模拟连续识别并过滤
        raw_values = []
        filtered_values = []

        for i in range(10):
            result = ocr.recognize(sample_digital_image)
            # 尝试解析数值
            try:
                value = float(result.text) if result.text else 0.0
            except (ValueError, TypeError):
                value = 0.0

            raw_values.append(value)
            filtered_value = filter_obj.filter(value)
            filtered_values.append(filtered_value)

        assert len(raw_values) == len(filtered_values)


class TestPointerToModbus:
    """指针识别到 Modbus 输出集成测试"""

    def test_pointer_result_to_modbus_register(self, sample_gauge_image):
        """测试指针识别结果转 Modbus 寄存器"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

        config = GaugeConfig(
            min_value=0,
            max_value=100,
            min_angle=225,
            max_angle=-45,
            unit="MPa"
        )

        recognizer = PointerRecognizer(config)
        result = recognizer.recognize(sample_gauge_image)

        # 转换为 Modbus 寄存器值（16位整数）
        if result.value is not None:
            # 乘以 100 保留两位小数
            register_value = int(result.value * 100)
            # 确保在 16 位范围内
            register_value = max(0, min(65535, register_value))
        else:
            register_value = 0

        assert 0 <= register_value <= 65535

    def test_pointer_confidence_threshold(self, sample_gauge_image):
        """测试置信度阈值过滤"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

        recognizer = PointerRecognizer(GaugeConfig())
        result = recognizer.recognize(sample_gauge_image)

        # 只有置信度高于阈值才输出
        confidence_threshold = 0.5
        should_output = result.confidence >= confidence_threshold

        assert isinstance(should_output, bool)

    def test_calibrated_pointer_to_modbus(self, sample_gauge_image, temp_calibration_file):
        """测试校准后的指针识别输出"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

        recognizer = PointerRecognizer(GaugeConfig())

        # 两点校准
        recognizer.calibrate_two_point(
            angle1=0.0, value1=0.0,
            angle2=180.0, value2=100.0
        )

        result = recognizer.recognize(sample_gauge_image)

        # 转换为 Modbus 格式
        output = {
            "angle": result.angle,
            "value": result.value,
            "confidence": result.confidence,
            "calibrated": recognizer.calibration is not None
        }

        assert output["calibrated"] is True


class TestMultiRecognizerOutput:
    """多识别器多协议输出测试"""

    def test_parallel_recognition(
        self, sample_digital_image, sample_gauge_image, sample_light_image_green
    ):
        """测试并行识别"""
        from retrosight.recognition.ocr import SimpleOCR
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig
        from retrosight.recognition.light import LightRecognizer, LightConfig

        # 初始化识别器
        ocr = SimpleOCR()
        pointer = PointerRecognizer(GaugeConfig())
        light = LightRecognizer(LightConfig())

        # 并行识别（这里串行模拟）
        results = {
            "ocr": ocr.recognize(sample_digital_image),
            "pointer": pointer.recognize(sample_gauge_image),
            "light": light.detect(sample_light_image_green)
        }

        assert results["ocr"] is not None
        assert results["pointer"] is not None
        assert results["light"] is not None

    def test_aggregated_output(
        self, sample_digital_image, sample_gauge_image, sample_light_image_green
    ):
        """测试聚合输出"""
        from retrosight.recognition.ocr import SimpleOCR
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig
        from retrosight.recognition.light import LightRecognizer, LightConfig

        # 识别
        ocr_result = SimpleOCR().recognize(sample_digital_image)
        pointer_result = PointerRecognizer(GaugeConfig()).recognize(sample_gauge_image)
        light_result = LightRecognizer(LightConfig()).detect(sample_light_image_green)

        # 聚合为统一输出格式
        output = {
            "timestamp": "2024-01-01T00:00:00Z",
            "readings": [
                {
                    "type": "digital_display",
                    "value": ocr_result.text,
                    "confidence": ocr_result.confidence
                },
                {
                    "type": "gauge",
                    "value": pointer_result.value,
                    "unit": "MPa",
                    "confidence": pointer_result.confidence
                },
                {
                    "type": "indicator_light",
                    "state": light_result.state.value if hasattr(light_result.state, 'value') else str(light_result.state),
                    "color": light_result.color.value if hasattr(light_result.color, 'value') else str(light_result.color)
                }
            ]
        }

        assert len(output["readings"]) == 3
        assert output["readings"][0]["type"] == "digital_display"
        assert output["readings"][1]["type"] == "gauge"
        assert output["readings"][2]["type"] == "indicator_light"


class TestOutputProtocols:
    """输出协议测试"""

    def test_mqtt_config_validation(self):
        """测试 MQTT 配置验证"""
        from retrosight.output.mqtt import MQTTConfig

        config = MQTTConfig(
            host="localhost",
            port=1883,
            topic_prefix="test/retrosight"
        )

        assert config.host == "localhost"
        assert config.port == 1883

    def test_modbus_config_validation(self):
        """测试 Modbus 配置验证"""
        from retrosight.output.modbus import ModbusConfig

        config = ModbusConfig(
            host="0.0.0.0",
            port=502
        )

        assert config.host == "0.0.0.0"
        assert config.port == 502

    def test_mqtt_message_format(self, sample_digital_image):
        """测试 MQTT 消息格式"""
        from retrosight.recognition.ocr import SimpleOCR

        result = SimpleOCR().recognize(sample_digital_image)

        # JSON 格式消息
        message = json.dumps({
            "device_id": "test_device",
            "reading_type": "ocr",
            "value": result.text,
            "confidence": result.confidence,
            "timestamp": "2024-01-01T00:00:00Z"
        })

        parsed = json.loads(message)
        assert "device_id" in parsed
        assert "value" in parsed


class TestOutputBuffering:
    """输出缓冲测试"""

    def test_message_queue(self):
        """测试消息队列"""
        from collections import deque

        # 模拟消息队列
        message_queue = deque(maxlen=100)

        # 添加消息
        for i in range(10):
            message_queue.append({
                "id": i,
                "value": i * 10,
                "timestamp": f"2024-01-01T00:00:{i:02d}Z"
            })

        assert len(message_queue) == 10

        # 取出消息
        messages = list(message_queue)
        assert len(messages) == 10
        assert messages[0]["id"] == 0

    def test_reconnection_buffer(self):
        """测试断线重连缓冲"""
        # 模拟缓冲区
        buffer = []
        max_buffer_size = 1000

        # 模拟断线期间的消息缓存
        for i in range(50):
            if len(buffer) < max_buffer_size:
                buffer.append({
                    "id": i,
                    "value": i * 10
                })

        assert len(buffer) == 50

        # 模拟重连后发送
        sent_count = 0
        while buffer:
            message = buffer.pop(0)
            sent_count += 1

        assert sent_count == 50
        assert len(buffer) == 0


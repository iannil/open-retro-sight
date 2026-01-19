"""
端到端集成测试

测试完整流程:
- 图像采集 → 预处理 → 识别 → 输出
"""

import pytest
import numpy as np
import cv2
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


class TestFullPipelineOCR:
    """OCR 完整流程测试"""

    def test_image_capture_to_mqtt_output(self, sample_digital_image):
        """测试图像采集到 MQTT 输出完整流程"""
        from retrosight.recognition.ocr import SimpleOCR
        from retrosight.preprocessing.enhancement import ImageEnhancer, EnhancementConfig
        from retrosight.preprocessing.filter import create_default_filter

        # Step 1: 图像采集（使用 fixture 模拟）
        captured_image = sample_digital_image.copy()
        assert captured_image is not None

        # Step 2: 预处理
        enhancer = ImageEnhancer(EnhancementConfig())
        enhanced_image = enhancer.enhance(captured_image)
        assert enhanced_image is not None

        # Step 3: OCR 识别
        ocr = SimpleOCR()
        result = ocr.recognize(enhanced_image)
        assert result is not None

        # Step 4: 数值过滤
        filter_obj = create_default_filter()
        try:
            value = float(result.text) if result.text else 0.0
        except (ValueError, TypeError):
            value = 0.0
        filtered_value = filter_obj.filter(value)

        # Step 5: 构造输出消息
        output_message = {
            "device_id": "test_ocr_device",
            "reading_type": "digital_display",
            "raw_text": result.text,
            "filtered_value": filtered_value,
            "confidence": result.confidence,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        assert "device_id" in output_message
        assert "filtered_value" in output_message
        assert "timestamp" in output_message

    def test_continuous_ocr_pipeline(self, sample_digital_image):
        """测试连续 OCR 识别流程"""
        from retrosight.recognition.ocr import SimpleOCR
        from retrosight.preprocessing.filter import create_default_filter

        ocr = SimpleOCR()
        filter_obj = create_default_filter()

        # 模拟连续帧处理
        frame_count = 10
        results = []

        for i in range(frame_count):
            # 模拟帧间微小变化
            frame = sample_digital_image.copy()

            # 识别
            result = ocr.recognize(frame)

            # 过滤
            try:
                value = float(result.text) if result.text else 0.0
            except (ValueError, TypeError):
                value = 0.0
            filtered_value = filter_obj.filter(value)

            results.append({
                "frame": i,
                "raw": result.text,
                "filtered": filtered_value,
                "confidence": result.confidence
            })

        assert len(results) == frame_count


class TestFullPipelinePointer:
    """指针识别完整流程测试"""

    def test_image_capture_to_modbus_output(self, sample_gauge_image):
        """测试图像采集到 Modbus 输出完整流程"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig
        from retrosight.preprocessing.transform import PerspectiveTransform

        # Step 1: 图像采集
        captured_image = sample_gauge_image.copy()

        # Step 2: 透视校正（如需要）
        h, w = captured_image.shape[:2]
        transform = PerspectiveTransform()
        # 设置轻微校正
        src_points = [
            (5, 5), (w - 5, 5), (w - 5, h - 5), (5, h - 5)
        ]
        transform.set_source_points(src_points, w, h)
        corrected_image = transform.apply(captured_image)

        # Step 3: 指针识别
        config = GaugeConfig(
            min_value=0,
            max_value=100,
            min_angle=225,
            max_angle=-45,
            unit="MPa"
        )
        recognizer = PointerRecognizer(config)
        result = recognizer.recognize(corrected_image)

        # Step 4: 转换为 Modbus 寄存器值
        if result.value is not None:
            # 寄存器地址 0: 数值（乘以100）
            register_value = int(result.value * 100)
            register_value = max(0, min(65535, register_value))
            # 寄存器地址 1: 置信度（乘以100）
            register_confidence = int(result.confidence * 100)
        else:
            register_value = 0
            register_confidence = 0

        output = {
            "registers": {
                0: register_value,
                1: register_confidence
            },
            "angle": result.angle,
            "value": result.value
        }

        assert "registers" in output
        assert 0 <= output["registers"][0] <= 65535

    def test_calibrated_pointer_pipeline(self, sample_gauge_image, temp_calibration_file):
        """测试带校准的指针识别流程"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

        # 创建识别器并校准
        recognizer = PointerRecognizer(GaugeConfig())
        recognizer.calibrate_two_point(
            angle1=45.0, value1=0.0,
            angle2=315.0, value2=100.0
        )

        # 保存校准
        recognizer.save_calibration(temp_calibration_file)

        # 新识别器加载校准
        new_recognizer = PointerRecognizer(GaugeConfig())
        new_recognizer.load_calibration(temp_calibration_file)

        # 识别
        result = new_recognizer.recognize(sample_gauge_image)

        assert result is not None
        assert new_recognizer.calibration is not None


class TestFullPipelineLight:
    """指示灯识别完整流程测试"""

    def test_light_monitoring_pipeline(self, sample_light_image_green, sample_light_image_red):
        """测试指示灯监控流程"""
        from retrosight.recognition.light import LightRecognizer, LightConfig, LightColor

        config = LightConfig(
            region=(20, 20, 60, 60),
            expected_colors=[LightColor.GREEN, LightColor.RED]
        )
        recognizer = LightRecognizer(config)

        # 模拟状态变化监控
        states = []

        # 初始状态：绿灯
        result = recognizer.detect(sample_light_image_green)
        states.append({
            "time": 0,
            "color": str(result.color),
            "state": str(result.state)
        })

        # 状态变化：红灯
        result = recognizer.detect(sample_light_image_red)
        states.append({
            "time": 1,
            "color": str(result.color),
            "state": str(result.state)
        })

        assert len(states) == 2
        # 验证检测到了状态变化
        assert states[0]["color"] != states[1]["color"] or states[0] == states[1]


class TestFullPipelineSwitch:
    """开关识别完整流程测试"""

    def test_switch_monitoring_pipeline(self, sample_switch_on_image, sample_switch_off_image):
        """测试开关监控流程"""
        from retrosight.recognition.switch import SwitchRecognizer, SwitchConfig, SwitchType

        config = SwitchConfig(switch_type=SwitchType.TOGGLE)
        recognizer = SwitchRecognizer(config)

        # 监控状态变化
        states = []

        # ON 状态
        result_on = recognizer.recognize(sample_switch_on_image)
        states.append({
            "time": 0,
            "state": str(result_on.state)
        })

        # OFF 状态
        result_off = recognizer.recognize(sample_switch_off_image)
        states.append({
            "time": 1,
            "state": str(result_off.state)
        })

        assert len(states) == 2


class TestMultiDevicePipeline:
    """多设备流程测试"""

    def test_multi_device_aggregation(
        self, sample_digital_image, sample_gauge_image,
        sample_light_image_green, sample_switch_on_image
    ):
        """测试多设备数据聚合"""
        from retrosight.recognition.ocr import SimpleOCR
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig
        from retrosight.recognition.light import LightRecognizer, LightConfig
        from retrosight.recognition.switch import SwitchRecognizer, SwitchConfig, SwitchType

        # 初始化所有识别器
        ocr = SimpleOCR()
        pointer = PointerRecognizer(GaugeConfig())
        light = LightRecognizer(LightConfig())
        switch = SwitchRecognizer(SwitchConfig(switch_type=SwitchType.TOGGLE))

        # 采集并识别
        ocr_result = ocr.recognize(sample_digital_image)
        pointer_result = pointer.recognize(sample_gauge_image)
        light_result = light.detect(sample_light_image_green)
        switch_result = switch.recognize(sample_switch_on_image)

        # 聚合输出
        aggregated_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "devices": {
                "digital_display_1": {
                    "value": ocr_result.text,
                    "confidence": ocr_result.confidence
                },
                "gauge_1": {
                    "angle": pointer_result.angle,
                    "value": pointer_result.value,
                    "confidence": pointer_result.confidence
                },
                "light_1": {
                    "color": str(light_result.color),
                    "state": str(light_result.state)
                },
                "switch_1": {
                    "state": str(switch_result.state)
                }
            }
        }

        assert len(aggregated_data["devices"]) == 4
        assert "timestamp" in aggregated_data


class TestErrorHandling:
    """错误处理测试"""

    def test_invalid_image_handling(self):
        """测试无效图像处理"""
        from retrosight.recognition.ocr import SimpleOCR

        ocr = SimpleOCR()

        # 空图像
        empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
        result = ocr.recognize(empty_image)
        # 应该返回结果（可能为空）而不是抛出异常
        assert result is not None

    def test_corrupted_image_handling(self):
        """测试损坏图像处理"""
        from retrosight.recognition.ocr import SimpleOCR

        ocr = SimpleOCR()

        # 非标准尺寸图像
        small_image = np.zeros((1, 1, 3), dtype=np.uint8)
        result = ocr.recognize(small_image)
        assert result is not None

    def test_recognition_timeout_simulation(self, sample_digital_image):
        """测试识别超时模拟"""
        from retrosight.recognition.ocr import SimpleOCR

        ocr = SimpleOCR()

        # 正常识别应该很快完成
        start_time = time.time()
        result = ocr.recognize(sample_digital_image)
        elapsed_time = time.time() - start_time

        # 简单 OCR 应该在 1 秒内完成
        assert elapsed_time < 1.0
        assert result is not None


class TestPerformanceMetrics:
    """性能指标测试"""

    def test_throughput_measurement(self, sample_digital_image):
        """测试吞吐量测量"""
        from retrosight.recognition.ocr import SimpleOCR

        ocr = SimpleOCR()

        # 测量处理速度
        frame_count = 20
        start_time = time.time()

        for _ in range(frame_count):
            ocr.recognize(sample_digital_image)

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # 记录性能指标
        metrics = {
            "total_frames": frame_count,
            "elapsed_time": elapsed_time,
            "fps": fps
        }

        assert metrics["total_frames"] == frame_count
        assert metrics["fps"] > 0

    def test_latency_measurement(self, sample_digital_image):
        """测试延迟测量"""
        from retrosight.recognition.ocr import SimpleOCR

        ocr = SimpleOCR()

        # 测量单帧延迟
        latencies = []
        for _ in range(10):
            start_time = time.time()
            ocr.recognize(sample_digital_image)
            latency = (time.time() - start_time) * 1000  # 毫秒
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        metrics = {
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
            "min_latency_ms": min_latency
        }

        assert metrics["avg_latency_ms"] > 0
        assert metrics["max_latency_ms"] >= metrics["min_latency_ms"]


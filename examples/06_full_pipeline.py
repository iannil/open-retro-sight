#!/usr/bin/env python3
"""
Open-RetroSight 示例 06: 完整流程示例

功能说明:
- 图像采集（模拟摄像头）
- 图像预处理（增强、去噪）
- 多类型识别（OCR、指针、指示灯、开关）
- 数据过滤（滑动平均、卡尔曼滤波）
- 数据输出（MQTT、Modbus）

依赖:
- opencv-python-headless
- numpy
- paho-mqtt (可选)
- pymodbus (可选)

运行方法:
    python examples/06_full_pipeline.py
"""

import numpy as np
import cv2
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DeviceReading:
    """设备读数"""
    device_id: str
    reading_type: str
    value: Any
    confidence: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimulatedCamera:
    """模拟摄像头"""

    def __init__(self, device_type: str = "ocr"):
        self.device_type = device_type
        self.frame_count = 0

    def read(self) -> tuple:
        """读取一帧图像"""
        self.frame_count += 1

        if self.device_type == "ocr":
            # 模拟数码管显示，数值随时间变化
            value = 100 + np.sin(self.frame_count * 0.1) * 50
            img = self._create_ocr_image(f"{value:.1f}")
        elif self.device_type == "gauge":
            # 模拟指针仪表
            angle = 45 + np.sin(self.frame_count * 0.05) * 90
            img = self._create_gauge_image(angle)
        elif self.device_type == "light":
            # 模拟指示灯状态变化
            colors = ["green", "green", "green", "yellow", "red"]
            color = colors[self.frame_count % len(colors)]
            img = self._create_light_image(color)
        else:
            img = np.zeros((100, 100, 3), dtype=np.uint8)

        return True, img

    def _create_ocr_image(self, text: str) -> np.ndarray:
        img = np.zeros((100, 250, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        # 添加轻微噪声
        noise = np.random.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def _create_gauge_image(self, angle: float) -> np.ndarray:
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        center = (100, 100)
        radius = 80
        cv2.circle(img, center, radius, (0, 0, 0), 2)
        angle_rad = np.radians(angle)
        tip_x = int(center[0] + radius * 0.75 * np.cos(angle_rad))
        tip_y = int(center[1] - radius * 0.75 * np.sin(angle_rad))
        cv2.line(img, center, (tip_x, tip_y), (255, 0, 0), 2)
        return img

    def _create_light_image(self, color: str) -> np.ndarray:
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        img[:] = (50, 50, 50)
        color_map = {"green": (0, 255, 0), "yellow": (0, 255, 255), "red": (0, 0, 255)}
        cv2.circle(img, (40, 40), 25, color_map.get(color, (100, 100, 100)), -1)
        return img


class DataBuffer:
    """数据缓冲区"""

    def __init__(self, max_size: int = 1000):
        self.buffer: List[DeviceReading] = []
        self.max_size = max_size

    def add(self, reading: DeviceReading):
        """添加读数"""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(reading)

    def get_all(self) -> List[DeviceReading]:
        """获取所有数据"""
        return list(self.buffer)

    def clear(self) -> List[DeviceReading]:
        """清空并返回所有数据"""
        data = list(self.buffer)
        self.buffer.clear()
        return data


class MockMQTTPublisher:
    """模拟 MQTT 发布器"""

    def __init__(self, topic_prefix: str = "retrosight"):
        self.topic_prefix = topic_prefix
        self.published_count = 0

    def publish(self, topic: str, payload: str):
        """发布消息"""
        self.published_count += 1
        # 实际应用中这里会发送到 MQTT Broker
        pass

    def get_stats(self) -> Dict[str, int]:
        return {"published": self.published_count}


class MockModbusServer:
    """模拟 Modbus 服务器"""

    def __init__(self):
        self.registers = [0] * 100
        self.update_count = 0

    def write_register(self, address: int, value: int):
        """写入寄存器"""
        if 0 <= address < len(self.registers):
            self.registers[address] = value & 0xFFFF
            self.update_count += 1

    def get_stats(self) -> Dict[str, int]:
        return {"updates": self.update_count}


def run_full_pipeline():
    """运行完整流程"""
    print("=" * 60)
    print("Open-RetroSight 完整流程示例")
    print("=" * 60)

    # 导入模块
    from retrosight.recognition.ocr import SimpleOCR
    from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig
    from retrosight.recognition.light import LightRecognizer, LightConfig
    from retrosight.preprocessing.enhancement import ImageEnhancer, EnhancementConfig
    from retrosight.preprocessing.filter import create_default_filter

    # 初始化组件
    print("\n1. 初始化组件...")

    # 识别器
    ocr = SimpleOCR()
    pointer = PointerRecognizer(GaugeConfig(min_value=0, max_value=100))
    light = LightRecognizer(LightConfig())
    print("   - 识别器: OCR, 指针, 指示灯")

    # 预处理
    enhancer = ImageEnhancer(EnhancementConfig(denoise=True))
    print("   - 预处理: 图像增强")

    # 数据过滤
    ocr_filter = create_default_filter()
    pointer_filter = create_default_filter()
    print("   - 数据滤波: 滑动平均")

    # 模拟设备
    cameras = {
        "device_ocr_1": SimulatedCamera("ocr"),
        "device_gauge_1": SimulatedCamera("gauge"),
        "device_light_1": SimulatedCamera("light")
    }
    print(f"   - 模拟设备: {len(cameras)} 个")

    # 输出
    mqtt = MockMQTTPublisher()
    modbus = MockModbusServer()
    buffer = DataBuffer()
    print("   - 输出: MQTT, Modbus, 缓冲区")

    # 处理循环
    print("\n2. 开始处理循环 (模拟 20 帧)...")
    print("-" * 60)

    stats = {
        "frames_processed": 0,
        "readings_generated": 0,
        "start_time": time.time()
    }

    for frame_idx in range(20):
        frame_readings = []

        # 处理每个设备
        for device_id, camera in cameras.items():
            ret, frame = camera.read()
            if not ret:
                continue

            # 预处理
            enhanced = enhancer.enhance(frame)

            # 识别
            if "ocr" in device_id:
                result = ocr.recognize(enhanced)
                try:
                    raw_value = float(result.text) if result.text else 0.0
                except ValueError:
                    raw_value = 0.0
                filtered_value = ocr_filter.filter(raw_value)

                reading = DeviceReading(
                    device_id=device_id,
                    reading_type="ocr",
                    value=filtered_value,
                    confidence=result.confidence,
                    timestamp=datetime.now().isoformat(),
                    metadata={"raw_text": result.text}
                )

                # 写入 Modbus
                modbus.write_register(0, int(filtered_value * 100))
                modbus.write_register(1, int(result.confidence * 100))

            elif "gauge" in device_id:
                result = pointer.recognize(enhanced)
                raw_value = result.value if result.value else 0.0
                filtered_value = pointer_filter.filter(raw_value)

                reading = DeviceReading(
                    device_id=device_id,
                    reading_type="gauge",
                    value=filtered_value,
                    confidence=result.confidence,
                    timestamp=datetime.now().isoformat(),
                    metadata={"angle": result.angle}
                )

                # 写入 Modbus
                modbus.write_register(10, int(filtered_value * 100))

            elif "light" in device_id:
                result = light.detect(enhanced)

                reading = DeviceReading(
                    device_id=device_id,
                    reading_type="light",
                    value=str(result.color),
                    confidence=1.0,
                    timestamp=datetime.now().isoformat(),
                    metadata={"state": str(result.state)}
                )

                # 写入 Modbus (状态编码)
                color_code = {"green": 1, "yellow": 2, "red": 3}.get(
                    str(result.color).lower().split('.')[-1], 0
                )
                modbus.write_register(20, color_code)

            else:
                continue

            frame_readings.append(reading)
            buffer.add(reading)

            # 发布 MQTT
            mqtt.publish(
                f"retrosight/{device_id}/{reading.reading_type}",
                json.dumps({
                    "value": reading.value,
                    "confidence": reading.confidence,
                    "timestamp": reading.timestamp
                }, default=str)
            )

        stats["frames_processed"] += 1
        stats["readings_generated"] += len(frame_readings)

        # 每 5 帧输出一次状态
        if (frame_idx + 1) % 5 == 0:
            print(f"   帧 {frame_idx + 1}: 生成 {len(frame_readings)} 条读数")

    # 统计
    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    stats["fps"] = stats["frames_processed"] / stats["duration"] if stats["duration"] > 0 else 0

    print("-" * 60)
    print("\n3. 处理完成，统计信息:")
    print(f"   - 处理帧数: {stats['frames_processed']}")
    print(f"   - 生成读数: {stats['readings_generated']}")
    print(f"   - 处理时间: {stats['duration']:.2f} 秒")
    print(f"   - 平均帧率: {stats['fps']:.1f} FPS")
    print(f"   - MQTT 发布: {mqtt.get_stats()['published']} 条")
    print(f"   - Modbus 更新: {modbus.get_stats()['updates']} 次")
    print(f"   - 缓冲区数据: {len(buffer.get_all())} 条")

    # 显示最后几条读数
    print("\n4. 最近的读数:")
    print("-" * 60)
    for reading in buffer.get_all()[-6:]:
        value_str = f"{reading.value:.2f}" if isinstance(reading.value, float) else str(reading.value)
        print(f"   [{reading.reading_type:6s}] {reading.device_id}: {value_str} (置信度: {reading.confidence:.0%})")


def architecture_overview():
    """架构概览"""
    print("\n" + "=" * 60)
    print("系统架构概览")
    print("=" * 60)

    architecture = """
    ┌─────────────────────────────────────────────────────────┐
    │                    Open-RetroSight                       │
    │                    完整处理流程                          │
    └─────────────────────────────────────────────────────────┘

    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  图像    │    │  预处理  │    │   识别   │    │  输出    │
    │  采集    │───▶│  增强    │───▶│  引擎    │───▶│  协议    │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ 摄像头   │    │ 去噪     │    │ OCR      │    │ MQTT     │
    │ RTSP     │    │ 锐化     │    │ 指针     │    │ Modbus   │
    │ 图片文件 │    │ 去反光   │    │ 指示灯   │    │ HTTP     │
    │ 视频流   │    │ 透视校正 │    │ 开关     │    │ 文件     │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                         │
                                         ▼
                                    ┌──────────┐
                                    │ 数据滤波 │
                                    │ 卡尔曼   │
                                    │ 滑动平均 │
                                    │ 指数平滑 │
                                    └──────────┘

    数据流:
    1. 采集模块从摄像头/文件读取图像
    2. 预处理模块增强图像质量
    3. 识别引擎提取数值/状态
    4. 滤波器平滑输出数据
    5. 输出模块发送到外部系统
    """
    print(architecture)


def configuration_example():
    """配置示例"""
    print("\n" + "=" * 60)
    print("配置示例")
    print("=" * 60)

    config = {
        "device_id": "production_line_1",
        "capture": {
            "type": "camera",
            "source": "/dev/video0",
            "fps": 10,
            "resolution": [640, 480]
        },
        "preprocessing": {
            "denoise": True,
            "sharpen": True,
            "glare_removal": False
        },
        "recognition": {
            "type": "ocr",
            "engine": "simple",
            "confidence_threshold": 0.7
        },
        "filter": {
            "type": "moving_average",
            "window_size": 5
        },
        "output": {
            "mqtt": {
                "enabled": True,
                "host": "localhost",
                "port": 1883,
                "topic": "retrosight/production_line_1/ocr"
            },
            "modbus": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 502,
                "register_address": 0
            }
        }
    }

    print("\n配置文件示例 (JSON):")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    # 架构概览
    architecture_overview()

    # 运行完整流程
    run_full_pipeline()

    # 配置示例
    configuration_example()

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)

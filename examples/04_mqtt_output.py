#!/usr/bin/env python3
"""
Open-RetroSight 示例 04: MQTT 数据输出

功能说明:
- 配置 MQTT 连接
- 发布识别结果到 MQTT Broker
- 演示不同的消息格式

依赖:
- paho-mqtt

运行方法:
    # 需要先启动 MQTT Broker（如 Mosquitto）
    python examples/04_mqtt_output.py

注意:
    如果没有 MQTT Broker，示例将以模拟模式运行
"""

import json
import time
from datetime import datetime


def check_mqtt_available() -> bool:
    """检查 MQTT 是否可用"""
    try:
        import paho.mqtt.client as mqtt
        return True
    except ImportError:
        return False


def mqtt_publish_example():
    """MQTT 发布示例"""
    print("=" * 50)
    print("Open-RetroSight MQTT 输出示例")
    print("=" * 50)

    if not check_mqtt_available():
        print("\n[警告] paho-mqtt 未安装，运行模拟模式")
        print("       安装: pip install paho-mqtt")
        mqtt_simulate_mode()
        return

    from retrosight.output.mqtt import MQTTPublisher, MQTTConfig

    # 配置 MQTT
    print("\n1. 配置 MQTT 连接...")
    config = MQTTConfig(
        host="localhost",
        port=1883,
        topic_prefix="retrosight/demo"
    )
    print(f"   Broker: {config.host}:{config.port}")
    print(f"   Topic: {config.topic_prefix}/#")

    # 创建发布器
    print("\n2. 创建 MQTTPublisher...")
    try:
        publisher = MQTTPublisher(config)
        print("   发布器创建成功")
    except Exception as e:
        print(f"   创建失败: {e}")
        print("\n   切换到模拟模式...")
        mqtt_simulate_mode()
        return

    # 模拟识别数据
    print("\n3. 发布识别数据...")

    readings = [
        {"type": "ocr", "value": "123.45", "confidence": 0.95},
        {"type": "gauge", "value": 67.8, "unit": "MPa", "confidence": 0.88},
        {"type": "light", "color": "green", "state": "on"},
    ]

    for reading in readings:
        topic = f"{config.topic_prefix}/{reading['type']}"
        message = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "device_id": "demo_device",
            **reading
        })

        try:
            publisher.publish(topic, message)
            print(f"   已发布: {topic}")
        except Exception as e:
            print(f"   发布失败: {e}")

    print("\n4. 关闭连接...")
    publisher.disconnect()
    print("   连接已关闭")


def mqtt_simulate_mode():
    """MQTT 模拟模式（无需实际 Broker）"""
    print("\n" + "=" * 50)
    print("MQTT 模拟模式")
    print("=" * 50)

    print("\n模拟 MQTT 消息发布:")
    print("-" * 60)

    # 模拟数据
    messages = [
        {
            "topic": "retrosight/device1/ocr",
            "payload": {
                "timestamp": "2024-01-01T10:00:00Z",
                "device_id": "device1",
                "value": "456.78",
                "confidence": 0.92
            }
        },
        {
            "topic": "retrosight/device1/gauge",
            "payload": {
                "timestamp": "2024-01-01T10:00:01Z",
                "device_id": "device1",
                "angle": 135.5,
                "value": 75.2,
                "unit": "bar",
                "confidence": 0.85
            }
        },
        {
            "topic": "retrosight/device1/light",
            "payload": {
                "timestamp": "2024-01-01T10:00:02Z",
                "device_id": "device1",
                "color": "green",
                "state": "on",
                "brightness": 0.95
            }
        }
    ]

    for msg in messages:
        print(f"\nTopic: {msg['topic']}")
        print(f"Payload: {json.dumps(msg['payload'], indent=2)}")

    print("\n" + "-" * 60)


def mqtt_with_recognition():
    """结合识别的 MQTT 输出示例"""
    print("\n" + "=" * 50)
    print("结合识别的 MQTT 输出示例")
    print("=" * 50)

    from retrosight.recognition.ocr import SimpleOCR
    import numpy as np
    import cv2

    # 创建测试图像
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, "789", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # OCR 识别
    print("\n1. OCR 识别...")
    ocr = SimpleOCR()
    result = ocr.recognize(img)
    print(f"   识别结果: {result.text} (置信度: {result.confidence:.2%})")

    # 构造 MQTT 消息
    print("\n2. 构造 MQTT 消息...")
    message = {
        "timestamp": datetime.now().isoformat(),
        "device_id": "test_ocr_device",
        "reading": {
            "type": "digital_display",
            "raw_text": result.text,
            "numeric_value": float(result.text) if result.text.replace('.', '').isdigit() else None,
            "confidence": result.confidence
        },
        "metadata": {
            "source": "Open-RetroSight",
            "version": "0.2.0"
        }
    }

    print("\n   消息内容:")
    print(json.dumps(message, indent=4, default=str))

    # 显示 Topic 结构
    print("\n3. 推荐的 Topic 结构:")
    print("   retrosight/")
    print("   ├── {device_id}/")
    print("   │   ├── ocr          # OCR 识别结果")
    print("   │   ├── gauge        # 指针仪表读数")
    print("   │   ├── light        # 指示灯状态")
    print("   │   ├── switch       # 开关状态")
    print("   │   └── status       # 设备状态")
    print("   └── system/")
    print("       ├── heartbeat    # 心跳")
    print("       └── alerts       # 告警")


def message_format_examples():
    """消息格式示例"""
    print("\n" + "=" * 50)
    print("MQTT 消息格式示例")
    print("=" * 50)

    # 不同类型的消息格式
    formats = {
        "简单格式": {
            "value": 123.45,
            "timestamp": "2024-01-01T10:00:00Z"
        },
        "完整格式": {
            "timestamp": "2024-01-01T10:00:00Z",
            "device_id": "device_001",
            "reading": {
                "type": "ocr",
                "value": "123.45",
                "confidence": 0.95
            },
            "quality": {
                "valid": True,
                "error_code": 0
            }
        },
        "批量格式": {
            "timestamp": "2024-01-01T10:00:00Z",
            "device_id": "device_001",
            "readings": [
                {"channel": 1, "value": 123.45, "type": "ocr"},
                {"channel": 2, "value": 67.8, "type": "gauge"},
                {"channel": 3, "value": "on", "type": "switch"}
            ]
        },
        "Sparkplug B 格式（简化）": {
            "timestamp": 1704067200000,
            "metrics": [
                {"name": "temperature", "value": 25.5, "type": "Float"},
                {"name": "pressure", "value": 101.3, "type": "Float"},
                {"name": "status", "value": True, "type": "Boolean"}
            ],
            "seq": 1
        }
    }

    for name, format_data in formats.items():
        print(f"\n{name}:")
        print("-" * 40)
        print(json.dumps(format_data, indent=2))


if __name__ == "__main__":
    # MQTT 发布示例
    mqtt_publish_example()

    # 结合识别的示例
    mqtt_with_recognition()

    # 消息格式示例
    message_format_examples()

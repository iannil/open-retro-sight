#!/usr/bin/env python3
"""
Open-RetroSight 示例 05: Modbus TCP 服务

功能说明:
- 启动 Modbus TCP Server
- 将识别结果写入保持寄存器
- 演示寄存器映射和数据类型转换

依赖:
- pymodbus

运行方法:
    python examples/05_modbus_server.py

注意:
    如果 pymodbus 未安装，示例将以模拟模式运行
"""

import struct
import time
from typing import Dict, Any


def check_modbus_available() -> bool:
    """检查 Modbus 是否可用"""
    try:
        import pymodbus
        return True
    except ImportError:
        return False


def modbus_server_example():
    """Modbus 服务器示例"""
    print("=" * 50)
    print("Open-RetroSight Modbus TCP 服务示例")
    print("=" * 50)

    if not check_modbus_available():
        print("\n[警告] pymodbus 未安装，运行模拟模式")
        print("       安装: pip install pymodbus")
        modbus_simulate_mode()
        return

    from retrosight.output.modbus import ModbusServer, ModbusConfig

    # 配置
    print("\n1. 配置 Modbus 服务器...")
    config = ModbusConfig(
        host="0.0.0.0",
        port=5020  # 使用非标准端口避免权限问题
    )
    print(f"   监听地址: {config.host}:{config.port}")

    # 寄存器映射
    print("\n2. 寄存器映射表:")
    print("-" * 60)
    print(f"{'地址':^10} | {'名称':^20} | {'类型':^15}")
    print("-" * 60)
    register_map = [
        (0, "OCR 数值 (x100)", "UINT16"),
        (1, "OCR 置信度 (x100)", "UINT16"),
        (2, "仪表角度 (x10)", "INT16"),
        (3, "仪表数值 (x100)", "UINT16"),
        (4, "仪表置信度 (x100)", "UINT16"),
        (5, "指示灯状态", "UINT16"),
        (6, "开关状态", "UINT16"),
        (7, "设备状态", "UINT16"),
    ]
    for addr, name, dtype in register_map:
        print(f"{addr:^10} | {name:^20} | {dtype:^15}")
    print("-" * 60)

    # 创建服务器（演示模式，不实际启动）
    print("\n3. 创建 Modbus 服务器...")
    try:
        server = ModbusServer(config)
        print("   服务器创建成功")

        # 写入模拟数据
        print("\n4. 写入模拟数据...")
        test_data = {
            0: 12345,  # OCR: 123.45
            1: 95,     # 置信度: 95%
            2: 1350,   # 角度: 135.0°
            3: 6780,   # 数值: 67.80
            4: 88,     # 置信度: 88%
            5: 1,      # 绿灯
            6: 1,      # 开关 ON
            7: 0,      # 正常
        }

        for addr, value in test_data.items():
            server.write_register(addr, value)
            print(f"   寄存器 {addr}: {value}")

        print("\n5. 服务器就绪（演示模式，未实际启动监听）")
        print("   实际使用时调用 server.start() 启动服务")

    except Exception as e:
        print(f"   创建失败: {e}")
        modbus_simulate_mode()


def modbus_simulate_mode():
    """Modbus 模拟模式"""
    print("\n" + "=" * 50)
    print("Modbus 模拟模式")
    print("=" * 50)

    # 模拟保持寄存器
    holding_registers = [0] * 100

    print("\n模拟写入识别数据到寄存器:")
    print("-" * 50)

    # OCR 数据
    ocr_value = 456.78
    ocr_confidence = 0.92
    holding_registers[0] = int(ocr_value * 100)  # 45678
    holding_registers[1] = int(ocr_confidence * 100)  # 92
    print(f"OCR 数值 (地址 0): {holding_registers[0]} (原值: {ocr_value})")
    print(f"OCR 置信度 (地址 1): {holding_registers[1]}% (原值: {ocr_confidence:.0%})")

    # 仪表数据
    gauge_angle = -45.5
    gauge_value = 87.65
    gauge_conf = 0.85
    # 角度可能为负，使用有符号整数
    holding_registers[2] = int(gauge_angle * 10) & 0xFFFF  # 转为无符号表示
    holding_registers[3] = int(gauge_value * 100)
    holding_registers[4] = int(gauge_conf * 100)
    print(f"仪表角度 (地址 2): {holding_registers[2]} (原值: {gauge_angle}°)")
    print(f"仪表数值 (地址 3): {holding_registers[3]} (原值: {gauge_value})")
    print(f"仪表置信度 (地址 4): {holding_registers[4]}% (原值: {gauge_conf:.0%})")

    # 状态数据
    light_status = 2  # 1=绿, 2=黄, 3=红
    switch_status = 1  # 0=OFF, 1=ON
    holding_registers[5] = light_status
    holding_registers[6] = switch_status
    print(f"指示灯状态 (地址 5): {holding_registers[5]} (1=绿, 2=黄, 3=红)")
    print(f"开关状态 (地址 6): {holding_registers[6]} (0=OFF, 1=ON)")

    print("-" * 50)


def data_type_conversion():
    """数据类型转换示例"""
    print("\n" + "=" * 50)
    print("数据类型转换示例")
    print("=" * 50)

    print("\n1. 整数缩放（常用方法）")
    print("-" * 40)

    float_values = [123.45, 0.001, -45.67, 9999.99]
    for val in float_values:
        scaled = int(val * 100)
        recovered = scaled / 100
        print(f"   原值: {val:10.2f} -> 缩放(x100): {scaled:8d} -> 还原: {recovered:10.2f}")

    print("\n2. IEEE 754 浮点数（32位）")
    print("-" * 40)

    float_values = [123.456, -789.012, 0.0001]
    for val in float_values:
        # 打包为32位浮点
        packed = struct.pack('>f', val)
        # 解包为两个16位整数
        high, low = struct.unpack('>HH', packed)
        # 还原
        repacked = struct.pack('>HH', high, low)
        recovered = struct.unpack('>f', repacked)[0]
        print(f"   原值: {val:12.4f} -> 高位: {high:5d}, 低位: {low:5d} -> 还原: {recovered:12.4f}")

    print("\n3. 状态编码")
    print("-" * 40)

    # 位域编码示例
    status_bits = {
        "running": True,
        "warning": False,
        "error": False,
        "maintenance": True
    }

    encoded = 0
    bit_map = {"running": 0, "warning": 1, "error": 2, "maintenance": 3}
    for name, value in status_bits.items():
        if value:
            encoded |= (1 << bit_map[name])

    print(f"   状态: {status_bits}")
    print(f"   编码: {encoded} (二进制: {bin(encoded)})")

    # 解码
    decoded = {}
    for name, bit in bit_map.items():
        decoded[name] = bool(encoded & (1 << bit))
    print(f"   解码: {decoded}")


def register_mapping_example():
    """寄存器映射示例"""
    print("\n" + "=" * 50)
    print("寄存器映射设计示例")
    print("=" * 50)

    # 示例映射表
    mapping = """
    ┌─────────┬──────────────────────┬──────────┬─────────────────┐
    │ 地址    │ 描述                 │ 数据类型 │ 比例因子        │
    ├─────────┼──────────────────────┼──────────┼─────────────────┤
    │ 0       │ 设备状态             │ UINT16   │ -               │
    │ 1       │ 错误代码             │ UINT16   │ -               │
    │ 2-3     │ 时间戳 (Unix秒)      │ UINT32   │ -               │
    ├─────────┼──────────────────────┼──────────┼─────────────────┤
    │ 10      │ 通道1 数值           │ INT16    │ x100            │
    │ 11      │ 通道1 置信度         │ UINT16   │ x100 (%)        │
    │ 12      │ 通道1 状态           │ UINT16   │ -               │
    ├─────────┼──────────────────────┼──────────┼─────────────────┤
    │ 20      │ 通道2 数值           │ INT16    │ x100            │
    │ 21      │ 通道2 置信度         │ UINT16   │ x100 (%)        │
    │ 22      │ 通道2 状态           │ UINT16   │ -               │
    ├─────────┼──────────────────────┼──────────┼─────────────────┤
    │ 100-109 │ 配置参数             │ -        │ -               │
    └─────────┴──────────────────────┴──────────┴─────────────────┘
    """
    print(mapping)

    print("状态码定义:")
    print("  0 = 正常")
    print("  1 = 低置信度")
    print("  2 = 无法识别")
    print("  3 = 传感器断开")
    print("  4 = 校准错误")


def with_recognition_example():
    """结合识别的 Modbus 示例"""
    print("\n" + "=" * 50)
    print("结合识别的 Modbus 数据写入示例")
    print("=" * 50)

    import numpy as np
    import cv2
    from retrosight.recognition.ocr import SimpleOCR
    from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

    # 模拟保持寄存器
    registers = [0] * 20

    # OCR 识别
    print("\n1. OCR 识别并写入寄存器...")
    img_ocr = np.zeros((100, 200, 3), dtype=np.uint8)
    img_ocr[:] = (40, 40, 40)
    cv2.putText(img_ocr, "567", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    ocr = SimpleOCR()
    ocr_result = ocr.recognize(img_ocr)

    try:
        ocr_value = float(ocr_result.text) if ocr_result.text else 0.0
    except ValueError:
        ocr_value = 0.0

    registers[0] = int(ocr_value * 100) & 0xFFFF
    registers[1] = int(ocr_result.confidence * 100)
    print(f"   OCR 值: {ocr_value} -> 寄存器[0] = {registers[0]}")
    print(f"   置信度: {ocr_result.confidence:.0%} -> 寄存器[1] = {registers[1]}")

    # 指针识别
    print("\n2. 指针识别并写入寄存器...")
    img_gauge = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.circle(img_gauge, (150, 150), 120, (0, 0, 0), 2)
    cv2.line(img_gauge, (150, 150), (240, 60), (255, 0, 0), 3)

    pointer = PointerRecognizer(GaugeConfig())
    pointer_result = pointer.recognize(img_gauge)

    angle_value = pointer_result.angle if pointer_result.angle else 0.0
    pointer_value = pointer_result.value if pointer_result.value else 0.0

    registers[10] = int(angle_value * 10) & 0xFFFF
    registers[11] = int(pointer_value * 100) & 0xFFFF
    registers[12] = int(pointer_result.confidence * 100)

    print(f"   角度: {angle_value:.1f}° -> 寄存器[10] = {registers[10]}")
    print(f"   数值: {pointer_value:.2f} -> 寄存器[11] = {registers[11]}")
    print(f"   置信度: {pointer_result.confidence:.0%} -> 寄存器[12] = {registers[12]}")

    # 显示寄存器状态
    print("\n3. 当前寄存器状态:")
    print("-" * 40)
    for i in range(15):
        if registers[i] != 0:
            print(f"   寄存器[{i:2d}] = {registers[i]}")


if __name__ == "__main__":
    # Modbus 服务器示例
    modbus_server_example()

    # 数据类型转换
    data_type_conversion()

    # 寄存器映射
    register_mapping_example()

    # 结合识别
    with_recognition_example()

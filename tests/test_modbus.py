"""
Modbus 模块单元测试
"""

import pytest
import struct
from unittest.mock import patch, MagicMock

from retrosight.output.modbus import (
    ModbusConfig,
    ModbusServer,
    ModbusClient,
    RegisterMapping,
    DataType,
    create_modbus_server,
)


class TestDataType:
    """数据类型测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert DataType.INT16.value == "int16"
        assert DataType.FLOAT32.value == "float32"
        assert DataType.FLOAT64.value == "float64"


class TestModbusConfig:
    """Modbus 配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = ModbusConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 502
        assert config.unit_id == 1
        assert config.max_registers == 1000

    def test_custom_values(self):
        """测试自定义值"""
        config = ModbusConfig(
            host="192.168.1.100",
            port=5020,
            unit_id=2
        )
        assert config.host == "192.168.1.100"
        assert config.port == 5020


class TestRegisterMapping:
    """寄存器映射测试"""

    def test_creation(self):
        """测试创建"""
        mapping = RegisterMapping(
            sensor_id="temp_01",
            address=0,
            data_type=DataType.FLOAT32,
            scale=10.0,
            description="温度传感器"
        )
        assert mapping.sensor_id == "temp_01"
        assert mapping.address == 0
        assert mapping.scale == 10.0

    def test_default_values(self):
        """测试默认值"""
        mapping = RegisterMapping(
            sensor_id="test",
            address=0,
            data_type=DataType.INT16
        )
        assert mapping.scale == 1.0
        assert mapping.offset == 0.0
        assert mapping.description == ""


class TestModbusServer:
    """Modbus 服务器测试"""

    def test_initialization(self):
        """测试初始化"""
        server = ModbusServer()
        assert not server._running
        assert len(server._mappings) == 0

    def test_add_mapping(self):
        """测试添加映射"""
        server = ModbusServer()
        mapping = RegisterMapping(
            sensor_id="temp_01",
            address=0,
            data_type=DataType.FLOAT32
        )
        server.add_mapping(mapping)

        assert "temp_01" in server._mappings

    def test_add_mapping_conflict(self):
        """测试地址冲突"""
        server = ModbusServer()

        mapping1 = RegisterMapping(
            sensor_id="sensor_1",
            address=0,
            data_type=DataType.FLOAT32  # 占用 2 个寄存器
        )
        server.add_mapping(mapping1)

        mapping2 = RegisterMapping(
            sensor_id="sensor_2",
            address=1,  # 与 sensor_1 冲突
            data_type=DataType.INT16
        )

        with pytest.raises(ValueError):
            server.add_mapping(mapping2)

    def test_remove_mapping(self):
        """测试移除映射"""
        server = ModbusServer()
        mapping = RegisterMapping(
            sensor_id="temp_01",
            address=0,
            data_type=DataType.FLOAT32
        )
        server.add_mapping(mapping)
        server.remove_mapping("temp_01")

        assert "temp_01" not in server._mappings

    def test_auto_map(self):
        """测试自动映射"""
        server = ModbusServer()
        sensor_ids = ["sensor_1", "sensor_2", "sensor_3"]
        server.auto_map(sensor_ids, DataType.FLOAT32, start_address=0)

        assert len(server._mappings) == 3
        # FLOAT32 占用 2 个寄存器
        assert server._mappings["sensor_1"].address == 0
        assert server._mappings["sensor_2"].address == 2
        assert server._mappings["sensor_3"].address == 4

    def test_value_to_registers_int16(self):
        """测试 INT16 转换"""
        server = ModbusServer()

        # 正数
        regs = server._value_to_registers(100, DataType.INT16)
        assert regs == [100]

        # 负数
        regs = server._value_to_registers(-100, DataType.INT16)
        assert regs == [65436]  # 补码表示

    def test_value_to_registers_float32(self):
        """测试 FLOAT32 转换"""
        server = ModbusServer()
        regs = server._value_to_registers(25.5, DataType.FLOAT32)

        assert len(regs) == 2

        # 验证转换回来
        value = server._registers_to_value(regs, DataType.FLOAT32)
        assert abs(value - 25.5) < 0.01

    def test_registers_to_value_float32(self):
        """测试 FLOAT32 逆转换"""
        server = ModbusServer()

        # 先转换为寄存器
        regs = server._value_to_registers(123.456, DataType.FLOAT32)

        # 再转换回来
        value = server._registers_to_value(regs, DataType.FLOAT32)
        assert abs(value - 123.456) < 0.001

    def test_get_register_size(self):
        """测试获取寄存器大小"""
        server = ModbusServer()

        assert server._get_register_size(DataType.INT16) == 1
        assert server._get_register_size(DataType.UINT16) == 1
        assert server._get_register_size(DataType.INT32) == 2
        assert server._get_register_size(DataType.FLOAT32) == 2
        assert server._get_register_size(DataType.FLOAT64) == 4

    def test_address_overlaps(self):
        """测试地址重叠检测"""
        server = ModbusServer()

        # 重叠
        assert server._address_overlaps(0, 2, 1, 2) is True

        # 不重叠
        assert server._address_overlaps(0, 2, 2, 2) is False

    def test_update_value(self):
        """测试更新值"""
        server = ModbusServer()
        mapping = RegisterMapping(
            sensor_id="temp_01",
            address=0,
            data_type=DataType.FLOAT32
        )
        server.add_mapping(mapping)
        server.update_value("temp_01", 25.5)

        # 验证值已更新
        value = server.get_value("temp_01")
        assert abs(value - 25.5) < 0.01

    def test_update_values_batch(self):
        """测试批量更新"""
        server = ModbusServer()
        server.auto_map(["s1", "s2"], DataType.FLOAT32)

        server.update_values({"s1": 10.0, "s2": 20.0})

        assert abs(server.get_value("s1") - 10.0) < 0.01
        assert abs(server.get_value("s2") - 20.0) < 0.01

    def test_get_register_map(self):
        """测试获取映射表"""
        server = ModbusServer()
        server.auto_map(["s1", "s2"], DataType.FLOAT32)

        reg_map = server.get_register_map()

        assert len(reg_map) == 2
        assert reg_map[0]["sensor_id"] == "s1"
        assert reg_map[1]["sensor_id"] == "s2"


class TestModbusClient:
    """Modbus 客户端测试"""

    def test_initialization(self):
        """测试初始化"""
        client = ModbusClient("localhost", 502)
        assert client.host == "localhost"
        assert client.port == 502
        assert client.unit_id == 1


class TestCreateModbusServer:
    """便捷函数测试"""

    def test_create_server(self):
        """测试创建服务器"""
        server = create_modbus_server(port=5020)
        assert server.config.port == 5020

    def test_create_server_with_mappings(self):
        """测试带映射创建"""
        mappings = [
            {
                "sensor_id": "temp",
                "address": 0,
                "data_type": "float32",
                "description": "温度"
            }
        ]
        server = create_modbus_server(mappings=mappings)

        assert "temp" in server._mappings

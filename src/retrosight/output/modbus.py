"""
Modbus TCP 输出模块

功能：
- Modbus TCP Server：伪装为 PLC 供 SCADA 系统抓取
- 寄存器映射：将传感器数据映射到 Modbus 寄存器
- 多数据类型支持：整数、浮点数、字符串
- 地址自动分配

基于 pymodbus 实现
"""

import threading
import logging
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import struct

logger = logging.getLogger(__name__)


class DataType(Enum):
    """数据类型"""
    INT16 = "int16"           # 16位有符号整数 (1个寄存器)
    UINT16 = "uint16"         # 16位无符号整数 (1个寄存器)
    INT32 = "int32"           # 32位有符号整数 (2个寄存器)
    UINT32 = "uint32"         # 32位无符号整数 (2个寄存器)
    FLOAT32 = "float32"       # 32位浮点数 (2个寄存器)
    FLOAT64 = "float64"       # 64位浮点数 (4个寄存器)


@dataclass
class ModbusConfig:
    """Modbus 配置"""
    host: str = "0.0.0.0"         # 监听地址
    port: int = 502               # 端口
    unit_id: int = 1              # 从站 ID
    max_registers: int = 1000     # 最大寄存器数量
    byte_order: str = "big"       # 字节序 ("big" 或 "little")
    word_order: str = "big"       # 字序 ("big" 或 "little")


@dataclass
class RegisterMapping:
    """寄存器映射"""
    sensor_id: str                # 传感器 ID
    address: int                  # 起始地址
    data_type: DataType           # 数据类型
    scale: float = 1.0            # 缩放系数
    offset: float = 0.0           # 偏移量
    description: str = ""         # 描述


class ModbusServer:
    """
    Modbus TCP 服务器

    将传感器数据通过 Modbus TCP 协议暴露给 SCADA 系统

    使用示例:
    ```python
    # 创建服务器
    server = ModbusServer(ModbusConfig(port=502))

    # 添加寄存器映射
    server.add_mapping(RegisterMapping(
        sensor_id="temp_01",
        address=0,
        data_type=DataType.FLOAT32,
        description="温度传感器1"
    ))

    # 启动服务器
    server.start()

    # 更新数据
    server.update_value("temp_01", 25.5)

    # 停止
    server.stop()
    ```
    """

    def __init__(self, config: Optional[ModbusConfig] = None):
        """
        初始化 Modbus 服务器

        Args:
            config: Modbus 配置
        """
        self.config = config or ModbusConfig()
        self._server = None
        self._context = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mappings: Dict[str, RegisterMapping] = {}
        self._registers: List[int] = [0] * self.config.max_registers
        self._lock = threading.Lock()

    def add_mapping(self, mapping: RegisterMapping):
        """
        添加寄存器映射

        Args:
            mapping: 寄存器映射配置
        """
        # 检查地址冲突
        size = self._get_register_size(mapping.data_type)
        for existing in self._mappings.values():
            existing_size = self._get_register_size(existing.data_type)
            if self._address_overlaps(
                mapping.address, size,
                existing.address, existing_size
            ):
                raise ValueError(
                    f"地址冲突: {mapping.sensor_id} 与 {existing.sensor_id}"
                )

        self._mappings[mapping.sensor_id] = mapping
        logger.info(f"添加映射: {mapping.sensor_id} -> 地址 {mapping.address}")

    def remove_mapping(self, sensor_id: str):
        """移除寄存器映射"""
        if sensor_id in self._mappings:
            del self._mappings[sensor_id]

    def auto_map(
        self,
        sensor_ids: List[str],
        data_type: DataType = DataType.FLOAT32,
        start_address: int = 0
    ):
        """
        自动分配寄存器地址

        Args:
            sensor_ids: 传感器 ID 列表
            data_type: 数据类型
            start_address: 起始地址
        """
        current_address = start_address
        size = self._get_register_size(data_type)

        for sensor_id in sensor_ids:
            mapping = RegisterMapping(
                sensor_id=sensor_id,
                address=current_address,
                data_type=data_type
            )
            self.add_mapping(mapping)
            current_address += size

    def update_value(self, sensor_id: str, value: float):
        """
        更新传感器值

        Args:
            sensor_id: 传感器 ID
            value: 数值
        """
        if sensor_id not in self._mappings:
            logger.warning(f"未知传感器: {sensor_id}")
            return

        mapping = self._mappings[sensor_id]

        # 应用缩放和偏移
        scaled_value = (value + mapping.offset) * mapping.scale

        # 转换为寄存器值
        registers = self._value_to_registers(scaled_value, mapping.data_type)

        # 写入寄存器
        with self._lock:
            for i, reg in enumerate(registers):
                addr = mapping.address + i
                if addr < len(self._registers):
                    self._registers[addr] = reg

        # 更新 Modbus 上下文
        self._update_context(mapping.address, registers)

    def update_values(self, values: Dict[str, float]):
        """
        批量更新传感器值

        Args:
            values: {sensor_id: value} 字典
        """
        for sensor_id, value in values.items():
            self.update_value(sensor_id, value)

    def get_value(self, sensor_id: str) -> Optional[float]:
        """
        获取传感器当前值

        Args:
            sensor_id: 传感器 ID

        Returns:
            当前值
        """
        if sensor_id not in self._mappings:
            return None

        mapping = self._mappings[sensor_id]
        size = self._get_register_size(mapping.data_type)

        with self._lock:
            registers = self._registers[mapping.address:mapping.address + size]

        value = self._registers_to_value(registers, mapping.data_type)

        # 反向应用缩放和偏移
        return (value / mapping.scale) - mapping.offset

    def start(self) -> bool:
        """
        启动 Modbus 服务器

        Returns:
            是否成功启动
        """
        if self._running:
            return True

        try:
            from pymodbus.server import StartTcpServer
            from pymodbus.datastore import (
                ModbusSequentialDataBlock,
                ModbusSlaveContext,
                ModbusServerContext
            )

            # 创建数据存储
            store = ModbusSlaveContext(
                di=ModbusSequentialDataBlock(0, [0] * self.config.max_registers),
                co=ModbusSequentialDataBlock(0, [0] * self.config.max_registers),
                hr=ModbusSequentialDataBlock(0, self._registers.copy()),
                ir=ModbusSequentialDataBlock(0, self._registers.copy())
            )

            self._context = ModbusServerContext(
                slaves={self.config.unit_id: store},
                single=False
            )

            # 在独立线程中启动服务器
            self._running = True
            self._thread = threading.Thread(
                target=self._server_thread,
                daemon=True
            )
            self._thread.start()

            logger.info(f"Modbus TCP 服务器已启动: {self.config.host}:{self.config.port}")
            return True

        except ImportError:
            logger.error("pymodbus 未安装，请运行: pip install pymodbus")
            return False
        except Exception as e:
            logger.error(f"Modbus 启动失败: {e}")
            return False

    def _server_thread(self):
        """服务器线程"""
        try:
            from pymodbus.server import StartTcpServer

            StartTcpServer(
                context=self._context,
                address=(self.config.host, self.config.port)
            )
        except Exception as e:
            logger.error(f"Modbus 服务器错误: {e}")
            self._running = False

    def stop(self):
        """停止 Modbus 服务器"""
        self._running = False
        # pymodbus 的同步服务器没有直接的停止方法
        # 服务器会在线程结束时自动清理
        logger.info("Modbus TCP 服务器已停止")

    def _update_context(self, address: int, values: List[int]):
        """更新 Modbus 上下文中的寄存器值"""
        if self._context is None:
            return

        try:
            slave = self._context[self.config.unit_id]
            # 更新保持寄存器
            slave.setValues(3, address, values)  # 3 = Holding Registers
            # 同时更新输入寄存器
            slave.setValues(4, address, values)  # 4 = Input Registers
        except Exception as e:
            logger.error(f"更新 Modbus 上下文失败: {e}")

    def _value_to_registers(self, value: float, data_type: DataType) -> List[int]:
        """将值转换为寄存器列表"""
        if data_type == DataType.INT16:
            val = int(value)
            val = max(-32768, min(32767, val))
            if val < 0:
                val = val + 65536
            return [val]

        elif data_type == DataType.UINT16:
            val = int(value)
            val = max(0, min(65535, val))
            return [val]

        elif data_type == DataType.INT32:
            val = int(value)
            packed = struct.pack('>i', val)
            return [
                struct.unpack('>H', packed[0:2])[0],
                struct.unpack('>H', packed[2:4])[0]
            ]

        elif data_type == DataType.UINT32:
            val = int(value)
            packed = struct.pack('>I', val)
            return [
                struct.unpack('>H', packed[0:2])[0],
                struct.unpack('>H', packed[2:4])[0]
            ]

        elif data_type == DataType.FLOAT32:
            packed = struct.pack('>f', value)
            return [
                struct.unpack('>H', packed[0:2])[0],
                struct.unpack('>H', packed[2:4])[0]
            ]

        elif data_type == DataType.FLOAT64:
            packed = struct.pack('>d', value)
            return [
                struct.unpack('>H', packed[0:2])[0],
                struct.unpack('>H', packed[2:4])[0],
                struct.unpack('>H', packed[4:6])[0],
                struct.unpack('>H', packed[6:8])[0]
            ]

        return [0]

    def _registers_to_value(self, registers: List[int], data_type: DataType) -> float:
        """将寄存器列表转换为值"""
        if not registers:
            return 0.0

        if data_type == DataType.INT16:
            val = registers[0]
            if val > 32767:
                val = val - 65536
            return float(val)

        elif data_type == DataType.UINT16:
            return float(registers[0])

        elif data_type == DataType.INT32:
            packed = struct.pack('>HH', registers[0], registers[1])
            return float(struct.unpack('>i', packed)[0])

        elif data_type == DataType.UINT32:
            packed = struct.pack('>HH', registers[0], registers[1])
            return float(struct.unpack('>I', packed)[0])

        elif data_type == DataType.FLOAT32:
            packed = struct.pack('>HH', registers[0], registers[1])
            return struct.unpack('>f', packed)[0]

        elif data_type == DataType.FLOAT64:
            packed = struct.pack('>HHHH', *registers[:4])
            return struct.unpack('>d', packed)[0]

        return 0.0

    def _get_register_size(self, data_type: DataType) -> int:
        """获取数据类型占用的寄存器数量"""
        sizes = {
            DataType.INT16: 1,
            DataType.UINT16: 1,
            DataType.INT32: 2,
            DataType.UINT32: 2,
            DataType.FLOAT32: 2,
            DataType.FLOAT64: 4
        }
        return sizes.get(data_type, 1)

    def _address_overlaps(
        self,
        addr1: int, size1: int,
        addr2: int, size2: int
    ) -> bool:
        """检查两个地址范围是否重叠"""
        return not (addr1 + size1 <= addr2 or addr2 + size2 <= addr1)

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running

    @property
    def mappings(self) -> Dict[str, RegisterMapping]:
        """获取所有映射"""
        return self._mappings.copy()

    def get_register_map(self) -> List[Dict[str, Any]]:
        """
        获取寄存器映射表

        Returns:
            映射表信息列表
        """
        result = []
        for sensor_id, mapping in self._mappings.items():
            size = self._get_register_size(mapping.data_type)
            result.append({
                "sensor_id": sensor_id,
                "address": mapping.address,
                "size": size,
                "data_type": mapping.data_type.value,
                "scale": mapping.scale,
                "offset": mapping.offset,
                "description": mapping.description
            })
        return sorted(result, key=lambda x: x["address"])


class ModbusClient:
    """
    Modbus TCP 客户端

    用于测试或从其他 Modbus 设备读取数据

    使用示例:
    ```python
    client = ModbusClient("192.168.1.100", 502)
    client.connect()

    # 读取保持寄存器
    value = client.read_float32(address=0)

    client.disconnect()
    ```
    """

    def __init__(self, host: str, port: int = 502, unit_id: int = 1):
        """
        初始化客户端

        Args:
            host: 服务器地址
            port: 端口
            unit_id: 从站 ID
        """
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self._client = None

    def connect(self) -> bool:
        """连接到服务器"""
        try:
            from pymodbus.client import ModbusTcpClient

            self._client = ModbusTcpClient(self.host, port=self.port)
            return self._client.connect()

        except ImportError:
            logger.error("pymodbus 未安装")
            return False
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self._client:
            self._client.close()
            self._client = None

    def read_registers(self, address: int, count: int) -> Optional[List[int]]:
        """读取保持寄存器"""
        if not self._client:
            return None

        result = self._client.read_holding_registers(
            address, count, slave=self.unit_id
        )

        if result.isError():
            return None

        return result.registers

    def read_float32(self, address: int) -> Optional[float]:
        """读取 32 位浮点数"""
        registers = self.read_registers(address, 2)
        if registers is None:
            return None

        packed = struct.pack('>HH', registers[0], registers[1])
        return struct.unpack('>f', packed)[0]

    def write_registers(self, address: int, values: List[int]) -> bool:
        """写入保持寄存器"""
        if not self._client:
            return False

        result = self._client.write_registers(
            address, values, slave=self.unit_id
        )

        return not result.isError()


def create_modbus_server(
    port: int = 502,
    mappings: Optional[List[Dict[str, Any]]] = None
) -> ModbusServer:
    """
    便捷函数：创建 Modbus 服务器

    Args:
        port: 端口
        mappings: 映射配置列表

    Returns:
        配置好的 Modbus 服务器
    """
    config = ModbusConfig(port=port)
    server = ModbusServer(config)

    if mappings:
        for m in mappings:
            mapping = RegisterMapping(
                sensor_id=m["sensor_id"],
                address=m.get("address", 0),
                data_type=DataType(m.get("data_type", "float32")),
                scale=m.get("scale", 1.0),
                offset=m.get("offset", 0.0),
                description=m.get("description", "")
            )
            server.add_mapping(mapping)

    return server

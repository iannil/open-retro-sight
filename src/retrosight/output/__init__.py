"""
输出模块

数据输出协议:
- MQTT: 发送到云端平台
- Modbus TCP: 伪装PLC供SCADA系统抓取
- Buffer: 断网续传缓存
"""

from retrosight.output.mqtt import (
    MQTTConfig,
    MQTTPublisher,
    MQTTSubscriber,
    SensorData,
    create_publisher,
)

from retrosight.output.modbus import (
    ModbusConfig,
    ModbusServer,
    ModbusClient,
    RegisterMapping,
    DataType,
    create_modbus_server,
)

from retrosight.output.buffer import (
    BufferConfig,
    BufferedMessage,
    PersistentBuffer,
    StoreAndForward,
    MemoryBuffer,
    Priority,
)

__all__ = [
    # mqtt
    "MQTTConfig",
    "MQTTPublisher",
    "MQTTSubscriber",
    "SensorData",
    "create_publisher",
    # modbus
    "ModbusConfig",
    "ModbusServer",
    "ModbusClient",
    "RegisterMapping",
    "DataType",
    "create_modbus_server",
    # buffer
    "BufferConfig",
    "BufferedMessage",
    "PersistentBuffer",
    "StoreAndForward",
    "MemoryBuffer",
    "Priority",
]

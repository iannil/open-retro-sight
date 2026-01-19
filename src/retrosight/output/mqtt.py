"""
MQTT 数据发布模块

支持功能：
- 连接 MQTT Broker
- 结构化 JSON 数据推送
- 断线自动重连
- QoS 等级配置
- 本地缓存（断网续传）

基于 paho-mqtt 实现
"""

import json
import time
import threading
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
from datetime import datetime
import queue

logger = logging.getLogger(__name__)


@dataclass
class MQTTConfig:
    """MQTT 配置"""
    host: str = "localhost"           # Broker 地址
    port: int = 1883                  # Broker 端口
    username: Optional[str] = None    # 用户名
    password: Optional[str] = None    # 密码
    client_id: str = "retrosight"     # 客户端ID
    keepalive: int = 60               # 心跳间隔（秒）
    qos: int = 1                      # QoS 等级 (0, 1, 2)
    retain: bool = False              # 是否保留消息
    topic_prefix: str = "retrosight"  # 主题前缀
    reconnect_delay: float = 5.0      # 重连延迟（秒）
    max_buffer_size: int = 1000       # 离线缓冲区大小
    use_tls: bool = False             # 是否使用 TLS
    ca_certs: Optional[str] = None    # CA 证书路径


@dataclass
class SensorData:
    """传感器数据"""
    sensor_id: str                    # 传感器ID
    value: float                      # 数值
    unit: str = ""                    # 单位
    confidence: float = 1.0           # 置信度
    timestamp: Optional[str] = None   # 时间戳 (ISO 格式)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(asdict(self), ensure_ascii=False)


class MQTTPublisher:
    """
    MQTT 数据发布器

    使用示例:
    ```python
    # 创建发布器
    publisher = MQTTPublisher(MQTTConfig(
        host="mqtt.example.com",
        topic_prefix="factory/line1"
    ))

    # 启动连接
    publisher.start()

    # 发布数据
    data = SensorData(
        sensor_id="temp_sensor_01",
        value=36.5,
        unit="°C",
        confidence=0.95
    )
    publisher.publish(data)

    # 或直接发布数值
    publisher.publish_value("pressure_01", 1.23, unit="MPa")

    # 停止
    publisher.stop()
    ```
    """

    def __init__(self, config: Optional[MQTTConfig] = None):
        """
        初始化 MQTT 发布器

        Args:
            config: MQTT 配置
        """
        self.config = config or MQTTConfig()
        self._client = None
        self._connected = False
        self._running = False
        self._buffer: deque = deque(maxlen=self.config.max_buffer_size)
        self._lock = threading.Lock()
        self._publish_queue: queue.Queue = queue.Queue()
        self._publish_thread: Optional[threading.Thread] = None
        self._on_connect_callbacks: List[Callable] = []
        self._on_disconnect_callbacks: List[Callable] = []

    def start(self) -> bool:
        """
        启动 MQTT 连接

        Returns:
            是否成功启动
        """
        if self._running:
            return True

        try:
            import paho.mqtt.client as mqtt

            # 创建客户端
            self._client = mqtt.Client(
                client_id=self.config.client_id,
                protocol=mqtt.MQTTv311
            )

            # 设置回调
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_publish = self._on_publish

            # 设置认证
            if self.config.username:
                self._client.username_pw_set(
                    self.config.username,
                    self.config.password
                )

            # 设置 TLS
            if self.config.use_tls:
                self._client.tls_set(ca_certs=self.config.ca_certs)

            # 连接
            self._client.connect_async(
                self.config.host,
                self.config.port,
                self.config.keepalive
            )

            # 启动网络循环
            self._client.loop_start()
            self._running = True

            # 启动发布线程
            self._publish_thread = threading.Thread(
                target=self._publish_loop,
                daemon=True
            )
            self._publish_thread.start()

            logger.info(f"MQTT 客户端已启动，连接到 {self.config.host}:{self.config.port}")
            return True

        except ImportError:
            logger.error("paho-mqtt 未安装，请运行: pip install paho-mqtt")
            return False
        except Exception as e:
            logger.error(f"MQTT 启动失败: {e}")
            return False

    def stop(self):
        """停止 MQTT 连接"""
        self._running = False

        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

        if self._publish_thread:
            self._publish_queue.put(None)  # 发送停止信号
            self._publish_thread.join(timeout=2.0)
            self._publish_thread = None

        self._connected = False
        logger.info("MQTT 客户端已停止")

    def publish(self, data: SensorData) -> bool:
        """
        发布传感器数据

        Args:
            data: 传感器数据

        Returns:
            是否成功加入发布队列
        """
        topic = f"{self.config.topic_prefix}/{data.sensor_id}"
        payload = data.to_json()
        return self._enqueue_message(topic, payload)

    def publish_value(
        self,
        sensor_id: str,
        value: float,
        unit: str = "",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        便捷方法：发布单个数值

        Args:
            sensor_id: 传感器ID
            value: 数值
            unit: 单位
            confidence: 置信度
            metadata: 额外元数据

        Returns:
            是否成功加入发布队列
        """
        data = SensorData(
            sensor_id=sensor_id,
            value=value,
            unit=unit,
            confidence=confidence,
            metadata=metadata or {}
        )
        return self.publish(data)

    def publish_raw(self, topic: str, payload: str) -> bool:
        """
        发布原始消息

        Args:
            topic: 主题（不含前缀）
            payload: 消息内容

        Returns:
            是否成功加入发布队列
        """
        full_topic = f"{self.config.topic_prefix}/{topic}"
        return self._enqueue_message(full_topic, payload)

    def publish_batch(self, data_list: List[SensorData]) -> int:
        """
        批量发布数据

        Args:
            data_list: 传感器数据列表

        Returns:
            成功加入队列的数量
        """
        count = 0
        for data in data_list:
            if self.publish(data):
                count += 1
        return count

    def _enqueue_message(self, topic: str, payload: str) -> bool:
        """将消息加入发布队列"""
        message = {"topic": topic, "payload": payload}

        if self._connected:
            try:
                self._publish_queue.put_nowait(message)
                return True
            except queue.Full:
                pass

        # 连接断开时缓存消息
        with self._lock:
            self._buffer.append(message)
            logger.debug(f"消息已缓存，当前缓冲区大小: {len(self._buffer)}")

        return True

    def _publish_loop(self):
        """发布循环（在独立线程中运行）"""
        while self._running:
            try:
                message = self._publish_queue.get(timeout=0.5)

                if message is None:  # 停止信号
                    break

                if self._connected and self._client:
                    self._client.publish(
                        message["topic"],
                        message["payload"],
                        qos=self.config.qos,
                        retain=self.config.retain
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"发布消息失败: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self._connected = True
            logger.info("MQTT 已连接")

            # 发送缓冲区中的消息
            self._flush_buffer()

            # 调用回调
            for callback in self._on_connect_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"连接回调执行失败: {e}")
        else:
            logger.error(f"MQTT 连接失败，返回码: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self._connected = False

        if rc != 0:
            logger.warning(f"MQTT 意外断开，返回码: {rc}")

        # 调用回调
        for callback in self._on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"断开回调执行失败: {e}")

    def _on_publish(self, client, userdata, mid):
        """发布完成回调"""
        logger.debug(f"消息已发布，mid: {mid}")

    def _flush_buffer(self):
        """发送缓冲区中的消息"""
        with self._lock:
            count = len(self._buffer)
            if count == 0:
                return

            logger.info(f"正在发送缓冲区中的 {count} 条消息...")

            while self._buffer and self._connected:
                message = self._buffer.popleft()
                try:
                    self._publish_queue.put_nowait(message)
                except queue.Full:
                    self._buffer.appendleft(message)
                    break

    def on_connect(self, callback: Callable):
        """
        注册连接回调

        Args:
            callback: 回调函数
        """
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable):
        """
        注册断开连接回调

        Args:
            callback: 回调函数
        """
        self._on_disconnect_callbacks.append(callback)

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    @property
    def buffer_size(self) -> int:
        """当前缓冲区大小"""
        with self._lock:
            return len(self._buffer)

    def __enter__(self) -> "MQTTPublisher":
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


class MQTTSubscriber:
    """
    MQTT 订阅器（用于接收命令/配置更新）

    使用示例:
    ```python
    subscriber = MQTTSubscriber(config)
    subscriber.subscribe("config/#", on_config_update)
    subscriber.start()
    ```
    """

    def __init__(self, config: Optional[MQTTConfig] = None):
        """初始化订阅器"""
        self.config = config or MQTTConfig()
        self._client = None
        self._running = False
        self._subscriptions: Dict[str, Callable] = {}

    def subscribe(self, topic: str, callback: Callable[[str, str], None]):
        """
        订阅主题

        Args:
            topic: 主题（相对于 topic_prefix）
            callback: 回调函数，签名为 (topic, payload)
        """
        full_topic = f"{self.config.topic_prefix}/{topic}"
        self._subscriptions[full_topic] = callback

    def start(self) -> bool:
        """启动订阅器"""
        if self._running:
            return True

        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(
                client_id=f"{self.config.client_id}_sub",
                protocol=mqtt.MQTTv311
            )

            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message

            if self.config.username:
                self._client.username_pw_set(
                    self.config.username,
                    self.config.password
                )

            if self.config.use_tls:
                self._client.tls_set(ca_certs=self.config.ca_certs)

            self._client.connect_async(
                self.config.host,
                self.config.port,
                self.config.keepalive
            )

            self._client.loop_start()
            self._running = True
            return True

        except Exception as e:
            logger.error(f"订阅器启动失败: {e}")
            return False

    def stop(self):
        """停止订阅器"""
        self._running = False
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            # 订阅所有已注册的主题
            for topic in self._subscriptions:
                client.subscribe(topic, self.config.qos)
                logger.info(f"已订阅主题: {topic}")

    def _on_message(self, client, userdata, msg):
        """消息回调"""
        topic = msg.topic
        payload = msg.payload.decode("utf-8")

        # 查找匹配的回调
        for pattern, callback in self._subscriptions.items():
            if self._topic_matches(pattern, topic):
                try:
                    callback(topic, payload)
                except Exception as e:
                    logger.error(f"消息处理失败: {e}")

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """检查主题是否匹配模式"""
        # 简单的通配符匹配
        if pattern == topic:
            return True

        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        for p, t in zip(pattern_parts, topic_parts):
            if p == "#":
                return True
            if p != "+" and p != t:
                return False

        return len(pattern_parts) == len(topic_parts)


def create_publisher(
    host: str = "localhost",
    port: int = 1883,
    topic_prefix: str = "retrosight"
) -> MQTTPublisher:
    """
    便捷函数：创建 MQTT 发布器

    Args:
        host: Broker 地址
        port: Broker 端口
        topic_prefix: 主题前缀

    Returns:
        配置好的 MQTT 发布器
    """
    config = MQTTConfig(
        host=host,
        port=port,
        topic_prefix=topic_prefix
    )
    return MQTTPublisher(config)

"""
断网续传缓存模块

功能：
- 本地持久化存储：SQLite / 文件
- 自动重试机制：连接恢复后自动补发
- 优先级队列：重要数据优先发送
- 过期清理：自动清理过期数据
- 容量管理：防止存储溢出

用于保证网络不稳定环境下的数据完整性
"""

import os
import json
import time
import sqlite3
import threading
import logging
from typing import Optional, Dict, Any, List, Callable, Generator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import queue

logger = logging.getLogger(__name__)


class Priority(Enum):
    """消息优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BufferConfig:
    """缓存配置"""
    storage_path: str = "data/buffer.db"   # 存储路径
    max_size_mb: float = 100.0             # 最大存储大小 (MB)
    max_age_hours: int = 24                # 最大保留时间 (小时)
    batch_size: int = 100                  # 批量发送大小
    retry_interval: float = 5.0            # 重试间隔 (秒)
    max_retries: int = 3                   # 最大重试次数
    auto_cleanup: bool = True              # 自动清理过期数据


@dataclass
class BufferedMessage:
    """缓存消息"""
    id: Optional[int] = None
    topic: str = ""
    payload: str = ""
    priority: Priority = Priority.NORMAL
    timestamp: Optional[str] = None
    retry_count: int = 0
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"


class PersistentBuffer:
    """
    持久化缓存

    基于 SQLite 的消息缓存，支持断网续传

    使用示例:
    ```python
    buffer = PersistentBuffer(BufferConfig(storage_path="data/buffer.db"))
    buffer.open()

    # 存储消息
    buffer.push(BufferedMessage(
        topic="sensors/temp",
        payload='{"value": 25.5}',
        priority=Priority.NORMAL
    ))

    # 获取待发送消息
    messages = buffer.pop_batch(batch_size=10)

    # 标记已发送
    for msg in messages:
        buffer.mark_sent(msg.id)

    buffer.close()
    ```
    """

    def __init__(self, config: Optional[BufferConfig] = None):
        """
        初始化缓存

        Args:
            config: 缓存配置
        """
        self.config = config or BufferConfig()
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def open(self):
        """打开数据库连接"""
        # 确保目录存在
        storage_path = Path(self.config.storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(storage_path),
            check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        logger.info(f"缓存数据库已打开: {self.config.storage_path}")

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self):
        """创建数据表"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    timestamp TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    sent_at TEXT
                )
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_priority_created
                ON messages (priority DESC, created_at ASC)
            ''')
            self._conn.commit()

    def push(self, message: BufferedMessage) -> int:
        """
        添加消息到缓存

        Args:
            message: 缓存消息

        Returns:
            消息 ID
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                INSERT INTO messages (topic, payload, priority, timestamp, retry_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                message.topic,
                message.payload,
                message.priority.value,
                message.timestamp,
                message.retry_count,
                message.created_at
            ))
            self._conn.commit()
            return cursor.lastrowid

    def push_batch(self, messages: List[BufferedMessage]) -> List[int]:
        """
        批量添加消息

        Args:
            messages: 消息列表

        Returns:
            消息 ID 列表
        """
        ids = []
        with self._lock:
            cursor = self._conn.cursor()
            for msg in messages:
                cursor.execute('''
                    INSERT INTO messages (topic, payload, priority, timestamp, retry_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    msg.topic,
                    msg.payload,
                    msg.priority.value,
                    msg.timestamp,
                    msg.retry_count,
                    msg.created_at
                ))
                ids.append(cursor.lastrowid)
            self._conn.commit()
        return ids

    def pop(self) -> Optional[BufferedMessage]:
        """
        获取一条待发送消息（按优先级）

        Returns:
            缓存消息，无消息返回 None
        """
        messages = self.pop_batch(1)
        return messages[0] if messages else None

    def pop_batch(self, batch_size: Optional[int] = None) -> List[BufferedMessage]:
        """
        批量获取待发送消息

        Args:
            batch_size: 批量大小，默认使用配置值

        Returns:
            消息列表
        """
        size = batch_size or self.config.batch_size

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT id, topic, payload, priority, timestamp, retry_count, created_at
                FROM messages
                WHERE sent_at IS NULL AND retry_count < ?
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
            ''', (self.config.max_retries, size))

            messages = []
            for row in cursor.fetchall():
                messages.append(BufferedMessage(
                    id=row['id'],
                    topic=row['topic'],
                    payload=row['payload'],
                    priority=Priority(row['priority']),
                    timestamp=row['timestamp'],
                    retry_count=row['retry_count'],
                    created_at=row['created_at']
                ))

            return messages

    def mark_sent(self, message_id: int):
        """
        标记消息已发送

        Args:
            message_id: 消息 ID
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                UPDATE messages SET sent_at = ? WHERE id = ?
            ''', (datetime.utcnow().isoformat() + "Z", message_id))
            self._conn.commit()

    def mark_failed(self, message_id: int):
        """
        标记消息发送失败（增加重试计数）

        Args:
            message_id: 消息 ID
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                UPDATE messages SET retry_count = retry_count + 1 WHERE id = ?
            ''', (message_id,))
            self._conn.commit()

    def delete(self, message_id: int):
        """
        删除消息

        Args:
            message_id: 消息 ID
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('DELETE FROM messages WHERE id = ?', (message_id,))
            self._conn.commit()

    def cleanup_sent(self):
        """清理已发送的消息"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('DELETE FROM messages WHERE sent_at IS NOT NULL')
            deleted = cursor.rowcount
            self._conn.commit()
            logger.info(f"清理已发送消息: {deleted} 条")

    def cleanup_expired(self):
        """清理过期消息"""
        cutoff = datetime.utcnow() - timedelta(hours=self.config.max_age_hours)
        cutoff_str = cutoff.isoformat() + "Z"

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                DELETE FROM messages WHERE created_at < ?
            ''', (cutoff_str,))
            deleted = cursor.rowcount
            self._conn.commit()

            if deleted > 0:
                logger.info(f"清理过期消息: {deleted} 条")

    def cleanup_failed(self):
        """清理超过最大重试次数的消息"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                DELETE FROM messages WHERE retry_count >= ?
            ''', (self.config.max_retries,))
            deleted = cursor.rowcount
            self._conn.commit()

            if deleted > 0:
                logger.warning(f"清理失败消息: {deleted} 条")

    def vacuum(self):
        """压缩数据库"""
        with self._lock:
            self._conn.execute('VACUUM')

    @property
    def pending_count(self) -> int:
        """待发送消息数量"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM messages
                WHERE sent_at IS NULL AND retry_count < ?
            ''', (self.config.max_retries,))
            return cursor.fetchone()[0]

    @property
    def total_count(self) -> int:
        """总消息数量"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM messages')
            return cursor.fetchone()[0]

    @property
    def storage_size_mb(self) -> float:
        """存储大小 (MB)"""
        try:
            size = os.path.getsize(self.config.storage_path)
            return size / (1024 * 1024)
        except OSError:
            return 0.0

    def __enter__(self):
        """上下文管理器入口"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


class StoreAndForward:
    """
    存储转发管理器

    自动管理缓存和重试逻辑

    使用示例:
    ```python
    def send_message(topic: str, payload: str) -> bool:
        # 实际发送逻辑
        return mqtt_client.publish(topic, payload)

    saf = StoreAndForward(
        send_func=send_message,
        config=BufferConfig()
    )
    saf.start()

    # 发送消息（自动缓存失败的消息）
    saf.send("sensors/temp", '{"value": 25.5}')

    saf.stop()
    ```
    """

    def __init__(
        self,
        send_func: Callable[[str, str], bool],
        config: Optional[BufferConfig] = None
    ):
        """
        初始化存储转发管理器

        Args:
            send_func: 发送函数，签名 (topic, payload) -> bool
            config: 缓存配置
        """
        self.send_func = send_func
        self.config = config or BufferConfig()
        self._buffer = PersistentBuffer(self.config)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue()
        self._online = True

    def start(self):
        """启动存储转发"""
        self._buffer.open()
        self._running = True

        # 启动后台线程处理缓存
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

        logger.info("存储转发已启动")

    def stop(self):
        """停止存储转发"""
        self._running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._buffer.close()
        logger.info("存储转发已停止")

    def send(
        self,
        topic: str,
        payload: str,
        priority: Priority = Priority.NORMAL
    ) -> bool:
        """
        发送消息（自动处理失败情况）

        Args:
            topic: 主题
            payload: 内容
            priority: 优先级

        Returns:
            是否立即发送成功
        """
        if self._online:
            try:
                if self.send_func(topic, payload):
                    return True
            except Exception as e:
                logger.warning(f"发送失败: {e}")
                self._online = False

        # 发送失败，存入缓存
        msg = BufferedMessage(
            topic=topic,
            payload=payload,
            priority=priority
        )
        self._buffer.push(msg)
        return False

    def set_online(self, online: bool):
        """
        设置在线状态

        Args:
            online: 是否在线
        """
        self._online = online
        if online:
            logger.info("连接恢复，开始发送缓存数据")

    def _worker_loop(self):
        """后台工作循环"""
        cleanup_interval = 3600  # 每小时清理一次
        last_cleanup = time.time()

        while self._running:
            try:
                # 定期清理
                if self.config.auto_cleanup:
                    if time.time() - last_cleanup > cleanup_interval:
                        self._buffer.cleanup_sent()
                        self._buffer.cleanup_expired()
                        self._buffer.cleanup_failed()
                        last_cleanup = time.time()

                # 如果在线且有缓存消息，尝试发送
                if self._online and self._buffer.pending_count > 0:
                    self._flush_buffer()

                time.sleep(self.config.retry_interval)

            except Exception as e:
                logger.error(f"工作循环错误: {e}")
                time.sleep(1)

    def _flush_buffer(self):
        """发送缓存中的消息"""
        messages = self._buffer.pop_batch()

        for msg in messages:
            try:
                if self.send_func(msg.topic, msg.payload):
                    self._buffer.mark_sent(msg.id)
                else:
                    self._buffer.mark_failed(msg.id)
                    self._online = False
                    break
            except Exception as e:
                logger.warning(f"发送缓存消息失败: {e}")
                self._buffer.mark_failed(msg.id)
                self._online = False
                break

    @property
    def pending_count(self) -> int:
        """待发送消息数量"""
        return self._buffer.pending_count

    @property
    def is_online(self) -> bool:
        """是否在线"""
        return self._online


class MemoryBuffer:
    """
    内存缓存

    轻量级的内存队列，适用于短暂断网场景

    使用示例:
    ```python
    buffer = MemoryBuffer(max_size=1000)
    buffer.push(message)
    messages = buffer.pop_batch(10)
    ```
    """

    def __init__(self, max_size: int = 1000):
        """
        初始化内存缓存

        Args:
            max_size: 最大消息数量
        """
        self.max_size = max_size
        self._queue: List[BufferedMessage] = []
        self._lock = threading.Lock()

    def push(self, message: BufferedMessage) -> bool:
        """
        添加消息

        Args:
            message: 缓存消息

        Returns:
            是否成功添加
        """
        with self._lock:
            if len(self._queue) >= self.max_size:
                # 移除最旧的低优先级消息
                self._queue.sort(key=lambda m: (m.priority.value, m.created_at))
                self._queue.pop(0)

            self._queue.append(message)
            # 按优先级和时间排序
            self._queue.sort(
                key=lambda m: (-m.priority.value, m.created_at)
            )
            return True

    def pop(self) -> Optional[BufferedMessage]:
        """获取一条消息"""
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            return None

    def pop_batch(self, batch_size: int) -> List[BufferedMessage]:
        """批量获取消息"""
        with self._lock:
            messages = self._queue[:batch_size]
            self._queue = self._queue[batch_size:]
            return messages

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._queue.clear()

    @property
    def count(self) -> int:
        """消息数量"""
        with self._lock:
            return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """是否为空"""
        return self.count == 0

    @property
    def is_full(self) -> bool:
        """是否已满"""
        return self.count >= self.max_size

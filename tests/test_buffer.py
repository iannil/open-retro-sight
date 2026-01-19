"""
缓存模块单元测试
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime

from retrosight.output.buffer import (
    BufferConfig,
    BufferedMessage,
    PersistentBuffer,
    StoreAndForward,
    MemoryBuffer,
    Priority,
)


class TestPriority:
    """优先级测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert Priority.LOW.value == 0
        assert Priority.NORMAL.value == 1
        assert Priority.HIGH.value == 2
        assert Priority.CRITICAL.value == 3

    def test_priority_comparison(self):
        """测试优先级比较"""
        assert Priority.CRITICAL.value > Priority.HIGH.value
        assert Priority.HIGH.value > Priority.NORMAL.value


class TestBufferConfig:
    """缓存配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = BufferConfig()
        assert config.max_size_mb == 100.0
        assert config.max_age_hours == 24
        assert config.batch_size == 100
        assert config.max_retries == 3

    def test_custom_values(self):
        """测试自定义值"""
        config = BufferConfig(
            storage_path="/tmp/test.db",
            max_size_mb=50.0,
            batch_size=50
        )
        assert config.storage_path == "/tmp/test.db"
        assert config.max_size_mb == 50.0


class TestBufferedMessage:
    """缓存消息测试"""

    def test_creation(self):
        """测试创建"""
        msg = BufferedMessage(
            topic="test/topic",
            payload='{"value": 1}',
            priority=Priority.HIGH
        )
        assert msg.topic == "test/topic"
        assert msg.priority == Priority.HIGH

    def test_auto_timestamp(self):
        """测试自动时间戳"""
        msg = BufferedMessage(topic="test", payload="data")
        assert msg.timestamp is not None
        assert msg.timestamp.endswith("Z")

    def test_default_values(self):
        """测试默认值"""
        msg = BufferedMessage(topic="test", payload="data")
        assert msg.priority == Priority.NORMAL
        assert msg.retry_count == 0


class TestPersistentBuffer:
    """持久化缓存测试"""

    @pytest.fixture
    def temp_db(self):
        """创建临时数据库"""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_open_close(self, temp_db):
        """测试打开关闭"""
        config = BufferConfig(storage_path=temp_db)
        buffer = PersistentBuffer(config)

        buffer.open()
        assert buffer._conn is not None

        buffer.close()
        assert buffer._conn is None

    def test_push_and_pop(self, temp_db):
        """测试推送和弹出"""
        config = BufferConfig(storage_path=temp_db)

        with PersistentBuffer(config) as buffer:
            msg = BufferedMessage(
                topic="test",
                payload="data"
            )
            msg_id = buffer.push(msg)
            assert msg_id > 0

            popped = buffer.pop()
            assert popped is not None
            assert popped.topic == "test"

    def test_push_batch(self, temp_db):
        """测试批量推送"""
        config = BufferConfig(storage_path=temp_db)

        with PersistentBuffer(config) as buffer:
            messages = [
                BufferedMessage(topic=f"topic_{i}", payload=f"data_{i}")
                for i in range(5)
            ]
            ids = buffer.push_batch(messages)

            assert len(ids) == 5
            assert buffer.pending_count == 5

    def test_pop_batch(self, temp_db):
        """测试批量弹出"""
        config = BufferConfig(storage_path=temp_db)

        with PersistentBuffer(config) as buffer:
            for i in range(10):
                buffer.push(BufferedMessage(topic=f"t{i}", payload="d"))

            batch = buffer.pop_batch(5)
            assert len(batch) == 5

    def test_priority_ordering(self, temp_db):
        """测试优先级排序"""
        config = BufferConfig(storage_path=temp_db)

        with PersistentBuffer(config) as buffer:
            # 按优先级倒序添加
            buffer.push(BufferedMessage(
                topic="low", payload="d", priority=Priority.LOW
            ))
            buffer.push(BufferedMessage(
                topic="high", payload="d", priority=Priority.HIGH
            ))
            buffer.push(BufferedMessage(
                topic="normal", payload="d", priority=Priority.NORMAL
            ))

            # 弹出应该按优先级顺序
            first = buffer.pop()
            assert first.topic == "high"

    def test_mark_sent(self, temp_db):
        """测试标记已发送"""
        config = BufferConfig(storage_path=temp_db)

        with PersistentBuffer(config) as buffer:
            msg_id = buffer.push(BufferedMessage(topic="t", payload="d"))
            assert buffer.pending_count == 1

            buffer.mark_sent(msg_id)
            assert buffer.pending_count == 0

    def test_mark_failed(self, temp_db):
        """测试标记失败"""
        config = BufferConfig(storage_path=temp_db, max_retries=3)

        with PersistentBuffer(config) as buffer:
            msg_id = buffer.push(BufferedMessage(topic="t", payload="d"))

            # 标记失败
            buffer.mark_failed(msg_id)
            buffer.mark_failed(msg_id)

            # 仍可弹出
            msg = buffer.pop()
            assert msg.retry_count == 2

            # 再次失败后超过重试限制
            buffer.mark_failed(msg_id)
            msg = buffer.pop()
            assert msg is None

    def test_cleanup_sent(self, temp_db):
        """测试清理已发送"""
        config = BufferConfig(storage_path=temp_db)

        with PersistentBuffer(config) as buffer:
            msg_id = buffer.push(BufferedMessage(topic="t", payload="d"))
            buffer.mark_sent(msg_id)

            assert buffer.total_count == 1

            buffer.cleanup_sent()
            assert buffer.total_count == 0

    def test_context_manager(self, temp_db):
        """测试上下文管理器"""
        config = BufferConfig(storage_path=temp_db)

        with PersistentBuffer(config) as buffer:
            buffer.push(BufferedMessage(topic="t", payload="d"))
            assert buffer.pending_count == 1


class TestMemoryBuffer:
    """内存缓存测试"""

    def test_initialization(self):
        """测试初始化"""
        buffer = MemoryBuffer(max_size=100)
        assert buffer.max_size == 100
        assert buffer.is_empty

    def test_push_and_pop(self):
        """测试推送和弹出"""
        buffer = MemoryBuffer()
        msg = BufferedMessage(topic="test", payload="data")

        buffer.push(msg)
        assert buffer.count == 1

        popped = buffer.pop()
        assert popped.topic == "test"
        assert buffer.is_empty

    def test_priority_ordering(self):
        """测试优先级排序"""
        buffer = MemoryBuffer()

        buffer.push(BufferedMessage(
            topic="low", payload="d", priority=Priority.LOW
        ))
        buffer.push(BufferedMessage(
            topic="high", payload="d", priority=Priority.HIGH
        ))

        first = buffer.pop()
        assert first.topic == "high"

    def test_max_size_overflow(self):
        """测试大小溢出"""
        buffer = MemoryBuffer(max_size=3)

        for i in range(5):
            buffer.push(BufferedMessage(topic=f"t{i}", payload="d"))

        assert buffer.count == 3

    def test_pop_batch(self):
        """测试批量弹出"""
        buffer = MemoryBuffer()

        for i in range(10):
            buffer.push(BufferedMessage(topic=f"t{i}", payload="d"))

        batch = buffer.pop_batch(5)
        assert len(batch) == 5
        assert buffer.count == 5

    def test_clear(self):
        """测试清空"""
        buffer = MemoryBuffer()
        buffer.push(BufferedMessage(topic="t", payload="d"))
        buffer.clear()

        assert buffer.is_empty

    def test_is_full(self):
        """测试是否已满"""
        buffer = MemoryBuffer(max_size=2)

        assert not buffer.is_full

        buffer.push(BufferedMessage(topic="t1", payload="d"))
        buffer.push(BufferedMessage(topic="t2", payload="d"))

        assert buffer.is_full


class TestStoreAndForward:
    """存储转发测试"""

    @pytest.fixture
    def temp_db(self):
        """创建临时数据库"""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_send_online(self, temp_db):
        """测试在线发送"""
        send_mock = Mock(return_value=True)
        config = BufferConfig(storage_path=temp_db)

        saf = StoreAndForward(send_func=send_mock, config=config)
        saf._buffer.open()
        saf._online = True

        result = saf.send("topic", "payload")

        assert result is True
        send_mock.assert_called_once_with("topic", "payload")

        saf._buffer.close()

    def test_send_offline(self, temp_db):
        """测试离线发送（缓存）"""
        send_mock = Mock(return_value=False)
        config = BufferConfig(storage_path=temp_db)

        saf = StoreAndForward(send_func=send_mock, config=config)
        saf._buffer.open()
        saf._online = True

        result = saf.send("topic", "payload")

        assert result is False
        assert saf.pending_count == 1

        saf._buffer.close()

    def test_set_online(self, temp_db):
        """测试设置在线状态"""
        config = BufferConfig(storage_path=temp_db)
        saf = StoreAndForward(send_func=Mock(), config=config)

        saf.set_online(True)
        assert saf.is_online

        saf.set_online(False)
        assert not saf.is_online

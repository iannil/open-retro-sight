"""
摄像头模块单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import queue

from retrosight.capture.camera import (
    Camera,
    CameraConfig,
    CameraType,
    capture_single_frame,
    list_cameras,
)


class TestCameraType:
    """摄像头类型测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert CameraType.USB.value == "usb"
        assert CameraType.CSI.value == "csi"
        assert CameraType.RTSP.value == "rtsp"
        assert CameraType.FILE.value == "file"


class TestCameraConfig:
    """摄像头配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = CameraConfig()
        assert config.source == 0
        assert config.width == 640
        assert config.height == 480
        assert config.fps == 30
        assert config.camera_type == CameraType.USB

    def test_custom_values(self):
        """测试自定义值"""
        config = CameraConfig(
            source=1,
            width=1280,
            height=720,
            fps=60,
            camera_type=CameraType.CSI
        )
        assert config.source == 1
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 60
        assert config.camera_type == CameraType.CSI

    def test_rtsp_source(self):
        """测试 RTSP 源"""
        config = CameraConfig(
            source="rtsp://192.168.1.100:554/stream",
            camera_type=CameraType.RTSP
        )
        assert config.source == "rtsp://192.168.1.100:554/stream"


class TestCamera:
    """摄像头类测试"""

    def test_initialization(self):
        """测试初始化"""
        camera = Camera()
        assert camera._running is False
        assert camera._cap is None
        assert camera.config.source == 0

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = CameraConfig(source=2, width=800, height=600)
        camera = Camera(config)
        assert camera.config.source == 2
        assert camera.config.width == 800

    @patch("cv2.VideoCapture")
    def test_start_success(self, mock_capture):
        """测试成功启动"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap

        camera = Camera()
        result = camera.start()

        assert result is True
        assert camera._running is True
        mock_cap.set.assert_called()

    @patch("cv2.VideoCapture")
    def test_start_failure(self, mock_capture):
        """测试启动失败"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        camera = Camera()
        result = camera.start()

        assert result is False
        assert camera._running is False

    @patch("cv2.VideoCapture")
    def test_stop(self, mock_capture):
        """测试停止"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap

        camera = Camera()
        camera.start()
        camera.stop()

        assert camera._running is False
        mock_cap.release.assert_called()

    def test_is_running_property(self):
        """测试 is_running 属性"""
        camera = Camera()
        assert camera.is_running is False

        camera._running = True
        assert camera.is_running is True

    @patch("cv2.VideoCapture")
    def test_resolution_property(self, mock_capture):
        """测试 resolution 属性"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            3: 1280.0,  # CAP_PROP_FRAME_WIDTH
            4: 720.0,   # CAP_PROP_FRAME_HEIGHT
        }.get(prop, 0)
        mock_capture.return_value = mock_cap

        camera = Camera()
        camera.start()
        width, height = camera.resolution

        assert width == 1280
        assert height == 720

    def test_resolution_when_not_started(self):
        """测试未启动时的分辨率"""
        config = CameraConfig(width=800, height=600)
        camera = Camera(config)

        assert camera.resolution == (800, 600)

    def test_read_when_not_running(self):
        """测试未运行时读取"""
        camera = Camera()
        result = camera.read(timeout=0.1)
        assert result is None

    def test_read_latest_when_no_frame(self):
        """测试无帧时读取最新帧"""
        camera = Camera()
        camera._running = True
        result = camera.read_latest()
        assert result is None

    def test_read_latest_with_frame(self):
        """测试有帧时读取最新帧"""
        camera = Camera()
        camera._running = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera._last_frame = test_frame

        result = camera.read_latest()
        assert result is not None
        assert result.shape == (480, 640, 3)

    @patch("cv2.VideoCapture")
    def test_context_manager(self, mock_capture):
        """测试上下文管理器"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap

        with Camera() as camera:
            assert camera._running is True

        # 退出后应该已停止
        mock_cap.release.assert_called()


class TestCaptureSingleFrame:
    """单帧捕获测试"""

    @patch("cv2.VideoCapture")
    def test_capture_success(self, mock_capture):
        """测试成功捕获"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap

        frame = capture_single_frame(0)

        assert frame is not None
        assert frame.shape == (480, 640, 3)
        mock_cap.release.assert_called()

    @patch("cv2.VideoCapture")
    def test_capture_failure_not_opened(self, mock_capture):
        """测试打开失败"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        frame = capture_single_frame(0)

        assert frame is None

    @patch("cv2.VideoCapture")
    def test_capture_failure_read(self, mock_capture):
        """测试读取失败"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_capture.return_value = mock_cap

        frame = capture_single_frame(0)

        assert frame is None


class TestListCameras:
    """列出摄像头测试"""

    @patch("cv2.VideoCapture")
    def test_list_cameras(self, mock_capture):
        """测试列出摄像头"""
        # 模拟：索引 0 和 2 可用，索引 1 不可用
        def create_mock_cap(index):
            mock = MagicMock()
            mock.isOpened.return_value = index in [0, 2]
            return mock

        mock_capture.side_effect = create_mock_cap

        cameras = list_cameras(max_index=5)

        assert 0 in cameras
        assert 1 not in cameras
        assert 2 in cameras

    @patch("cv2.VideoCapture")
    def test_list_cameras_none_available(self, mock_capture):
        """测试无可用摄像头"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap

        cameras = list_cameras(max_index=3)

        assert len(cameras) == 0


class TestCameraFrameQueue:
    """帧队列测试"""

    def test_frame_queue_initialization(self):
        """测试帧队列初始化"""
        config = CameraConfig(buffer_size=5)
        camera = Camera(config)

        assert camera._frame_queue.maxsize == 5

    @patch("cv2.VideoCapture")
    def test_frames_generator(self, mock_capture):
        """测试帧生成器"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap

        camera = Camera()
        camera._running = True

        # 预填充队列
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera._frame_queue.put(test_frame)
        camera._frame_queue.put(test_frame)

        # 获取两帧后停止
        frames_gen = camera.frames()
        count = 0
        for frame in frames_gen:
            count += 1
            if count >= 2:
                camera._running = False
                break

        assert count == 2

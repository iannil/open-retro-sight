"""
摄像头视频流采集模块

支持多种视频源：
- USB 摄像头
- CSI 摄像头 (Raspberry Pi)
- RTSP 视频流
- 图片文件 (测试用)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator, Union
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time


class CameraType(Enum):
    """摄像头类型"""
    USB = "usb"           # USB 摄像头
    CSI = "csi"           # CSI 摄像头 (树莓派)
    RTSP = "rtsp"         # RTSP 视频流
    FILE = "file"         # 图片/视频文件


@dataclass
class CameraConfig:
    """摄像头配置"""
    source: Union[int, str] = 0          # 视频源 (设备ID或URL/路径)
    width: int = 640                      # 分辨率宽度
    height: int = 480                     # 分辨率高度
    fps: int = 30                         # 帧率
    camera_type: CameraType = CameraType.USB
    auto_exposure: bool = True            # 自动曝光
    buffer_size: int = 2                  # 帧缓冲区大小


class Camera:
    """
    摄像头视频流采集类

    使用示例:
    ```python
    # USB 摄像头
    camera = Camera(CameraConfig(source=0))
    camera.start()

    for frame in camera.frames():
        # 处理帧
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()

    # 或使用上下文管理器
    with Camera(CameraConfig(source=0)) as camera:
        frame = camera.read()
    ```
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        """
        初始化摄像头

        Args:
            config: 摄像头配置，默认使用 USB 摄像头 0
        """
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.config.buffer_size)
        self._last_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """
        启动摄像头

        Returns:
            是否成功启动
        """
        if self._running:
            return True

        # 打开视频源
        self._cap = cv2.VideoCapture(self.config.source)

        if not self._cap.isOpened():
            return False

        # 设置分辨率
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        # 设置缓冲区大小（减少延迟）
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 启动采集线程
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def stop(self):
        """停止摄像头"""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # 清空队列
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def _capture_loop(self):
        """帧采集循环（在独立线程中运行）"""
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            # 更新最新帧
            with self._lock:
                self._last_frame = frame

            # 放入队列（如果满了就丢弃旧帧）
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        读取一帧图像

        Args:
            timeout: 超时时间（秒）

        Returns:
            图像帧，失败返回 None
        """
        if not self._running:
            return None

        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            # 如果队列为空，返回最后一帧
            with self._lock:
                return self._last_frame

    def read_latest(self) -> Optional[np.ndarray]:
        """
        读取最新一帧（跳过缓冲区中的旧帧）

        Returns:
            最新图像帧
        """
        with self._lock:
            return self._last_frame.copy() if self._last_frame is not None else None

    def frames(self, skip_frames: int = 0) -> Generator[np.ndarray, None, None]:
        """
        帧生成器

        Args:
            skip_frames: 每隔多少帧返回一次（用于降低处理频率）

        Yields:
            图像帧
        """
        frame_count = 0
        while self._running:
            frame = self.read()
            if frame is None:
                continue

            frame_count += 1
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue

            yield frame

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running

    @property
    def resolution(self) -> Tuple[int, int]:
        """当前分辨率 (宽, 高)"""
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (self.config.width, self.config.height)

    @property
    def fps(self) -> float:
        """当前帧率"""
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return self.config.fps

    def __enter__(self) -> "Camera":
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()

    def __del__(self):
        """析构函数"""
        self.stop()


def capture_single_frame(source: Union[int, str] = 0) -> Optional[np.ndarray]:
    """
    捕获单帧图像（便捷函数）

    Args:
        source: 视频源

    Returns:
        图像帧
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def list_cameras(max_index: int = 10) -> list:
    """
    列出可用的摄像头

    Args:
        max_index: 最大检测索引

    Returns:
        可用摄像头索引列表
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

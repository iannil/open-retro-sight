"""
图像采集模块

负责从摄像头获取视频流，支持:
- USB 摄像头
- CSI 摄像头 (Raspberry Pi)
- RTSP 视频流
- 图片文件 (测试用)
"""

from retrosight.capture.camera import (
    Camera,
    CameraConfig,
    CameraType,
    capture_single_frame,
    list_cameras,
)

__all__ = [
    "Camera",
    "CameraConfig",
    "CameraType",
    "capture_single_frame",
    "list_cameras",
]

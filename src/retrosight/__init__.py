"""
Open-RetroSight - 非侵入式工业边缘AI网关

通过计算机视觉将传统"哑设备"的数据数字化。
核心价值：不拆机、不停产、不改线，给老机器装上"数字眼睛"。
"""

__version__ = "0.1.0"
__author__ = "Open-RetroSight Team"

from retrosight.capture import Camera
from retrosight.recognition import OCRRecognizer
from retrosight.output import MQTTPublisher

__all__ = [
    "Camera",
    "OCRRecognizer",
    "MQTTPublisher",
    "__version__",
]

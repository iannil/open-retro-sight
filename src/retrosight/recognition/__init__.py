"""
识别模块

提供多种识别能力:
- OCR: 七段数码管/LCD屏幕数字识别
- Pointer: 指针式仪表识别
- Light: 状态指示灯识别
- Switch: 开关/旋钮位置识别
"""

from retrosight.recognition.ocr import (
    OCRRecognizer,
    OCRConfig,
    OCRResult,
    SimpleOCR,
    recognize_digits,
)

from retrosight.recognition.pointer import (
    PointerRecognizer,
    PointerResult,
    GaugeConfig,
    GaugeType,
    CalibrationPoint,
    CalibrationData,
    recognize_gauge,
)

from retrosight.recognition.light import (
    LightRecognizer,
    LightResult,
    LightConfig,
    LightColor,
    LightState,
    AndonMonitor,
    detect_light,
    detect_andon,
)

from retrosight.recognition.switch import (
    SwitchRecognizer,
    SwitchResult,
    SwitchConfig,
    SwitchType,
    SwitchState,
    MultiSwitchMonitor,
    detect_switch,
    detect_rotary,
)

__all__ = [
    # ocr
    "OCRRecognizer",
    "OCRConfig",
    "OCRResult",
    "SimpleOCR",
    "recognize_digits",
    # pointer
    "PointerRecognizer",
    "PointerResult",
    "GaugeConfig",
    "GaugeType",
    "CalibrationPoint",
    "CalibrationData",
    "recognize_gauge",
    # light
    "LightRecognizer",
    "LightResult",
    "LightConfig",
    "LightColor",
    "LightState",
    "AndonMonitor",
    "detect_light",
    "detect_andon",
    # switch
    "SwitchRecognizer",
    "SwitchResult",
    "SwitchConfig",
    "SwitchType",
    "SwitchState",
    "MultiSwitchMonitor",
    "detect_switch",
    "detect_rotary",
]

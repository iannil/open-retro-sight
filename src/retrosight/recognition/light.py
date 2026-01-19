"""
指示灯识别模块

功能：
- 状态指示灯检测：识别红/黄/绿等颜色
- Andon 灯识别：三色灯塔状态判断
- 闪烁检测：区分常亮和闪烁状态
- 多灯联动：同时监控多个指示灯

用于设备状态监控和 OEE 计算
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class LightColor(Enum):
    """指示灯颜色"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    WHITE = "white"
    ORANGE = "orange"
    UNKNOWN = "unknown"


class LightState(Enum):
    """指示灯状态"""
    OFF = "off"           # 熄灭
    ON = "on"             # 常亮
    BLINKING = "blinking" # 闪烁
    UNKNOWN = "unknown"   # 未知


@dataclass
class ColorRange:
    """HSV 颜色范围"""
    lower: Tuple[int, int, int]  # H, S, V 下限
    upper: Tuple[int, int, int]  # H, S, V 上限


@dataclass
class LightConfig:
    """指示灯配置"""
    # 检测区域 (x, y, width, height)，None 表示全图
    region: Optional[Tuple[int, int, int, int]] = None
    # 最小亮度阈值
    brightness_threshold: int = 100
    # 最小面积（像素）
    min_area: int = 50
    # 最大面积（像素）
    max_area: int = 10000
    # 闪烁检测窗口（帧数）
    blink_window: int = 30
    # 闪烁判定阈值（状态变化次数）
    blink_threshold: int = 2
    # 自定义颜色范围
    custom_colors: Dict[str, ColorRange] = field(default_factory=dict)


@dataclass
class LightResult:
    """指示灯识别结果"""
    color: LightColor             # 颜色
    state: LightState             # 状态
    brightness: float = 0.0       # 亮度 (0-255)
    confidence: float = 0.0       # 置信度
    position: Optional[Tuple[int, int]] = None  # 中心位置
    area: int = 0                 # 面积
    blink_frequency: float = 0.0  # 闪烁频率 (Hz)


# 默认 HSV 颜色范围
DEFAULT_COLOR_RANGES = {
    LightColor.RED: [
        ColorRange((0, 100, 100), (10, 255, 255)),    # 红色范围1
        ColorRange((160, 100, 100), (180, 255, 255))  # 红色范围2
    ],
    LightColor.YELLOW: [
        ColorRange((15, 100, 100), (35, 255, 255))
    ],
    LightColor.ORANGE: [
        ColorRange((10, 100, 100), (20, 255, 255))
    ],
    LightColor.GREEN: [
        ColorRange((35, 100, 100), (85, 255, 255))
    ],
    LightColor.BLUE: [
        ColorRange((85, 100, 100), (130, 255, 255))
    ],
    LightColor.WHITE: [
        ColorRange((0, 0, 200), (180, 30, 255))
    ]
}


class LightRecognizer:
    """
    指示灯识别器

    使用示例:
    ```python
    # 基本使用
    recognizer = LightRecognizer()
    result = recognizer.detect(image)
    print(f"颜色: {result.color}, 状态: {result.state}")

    # 配置特定区域
    config = LightConfig(region=(100, 100, 50, 50))
    recognizer = LightRecognizer(config)

    # 检测多个灯
    results = recognizer.detect_multiple(image)
    for r in results:
        print(f"{r.color}: {r.state}")
    ```
    """

    def __init__(self, config: Optional[LightConfig] = None):
        """
        初始化指示灯识别器

        Args:
            config: 指示灯配置
        """
        self.config = config or LightConfig()
        self._history: deque = deque(maxlen=self.config.blink_window)
        self._last_state: Optional[LightState] = None
        self._state_changes: int = 0
        self._last_change_time: float = 0

    def detect(self, image: np.ndarray) -> LightResult:
        """
        检测图像中的指示灯

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            指示灯识别结果
        """
        # 提取检测区域
        roi = self._extract_roi(image)

        # 检测颜色和亮度
        color, brightness, mask = self._detect_color(roi)

        # 检测位置和面积
        position, area = self._find_light_position(mask)

        # 确定状态（亮/灭）
        is_on = brightness > self.config.brightness_threshold and area >= self.config.min_area

        # 更新历史记录用于闪烁检测
        self._history.append(is_on)

        # 检测闪烁
        state, blink_freq = self._detect_blinking(is_on)

        # 计算置信度
        confidence = self._calculate_confidence(brightness, area, mask)

        return LightResult(
            color=color if is_on else LightColor.UNKNOWN,
            state=state,
            brightness=brightness,
            confidence=confidence,
            position=position,
            area=area,
            blink_frequency=blink_freq
        )

    def detect_multiple(
        self,
        image: np.ndarray,
        min_distance: int = 30
    ) -> List[LightResult]:
        """
        检测图像中的多个指示灯

        Args:
            image: 输入图像
            min_distance: 灯之间的最小距离

        Returns:
            指示灯结果列表
        """
        roi = self._extract_roi(image)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        results = []

        # 对每种颜色进行检测
        for color, ranges in DEFAULT_COLOR_RANGES.items():
            mask = self._create_color_mask(hsv, ranges)

            # 查找轮廓
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                if self.config.min_area <= area <= self.config.max_area:
                    # 计算中心点
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 检查是否与已检测到的灯太近
                        too_close = False
                        for r in results:
                            if r.position:
                                dist = np.sqrt(
                                    (cx - r.position[0])**2 +
                                    (cy - r.position[1])**2
                                )
                                if dist < min_distance:
                                    too_close = True
                                    break

                        if not too_close:
                            # 计算亮度
                            light_mask = np.zeros(mask.shape, dtype=np.uint8)
                            cv2.drawContours(light_mask, [contour], -1, 255, -1)
                            brightness = cv2.mean(
                                cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                                mask=light_mask
                            )[0]

                            results.append(LightResult(
                                color=color,
                                state=LightState.ON if brightness > self.config.brightness_threshold else LightState.OFF,
                                brightness=brightness,
                                confidence=min(1.0, area / 500),
                                position=(cx, cy),
                                area=int(area)
                            ))

        return results

    def detect_andon(self, image: np.ndarray) -> Dict[str, LightResult]:
        """
        检测 Andon 三色灯塔

        Args:
            image: 输入图像

        Returns:
            {"red": result, "yellow": result, "green": result}
        """
        results = self.detect_multiple(image)

        andon = {
            "red": None,
            "yellow": None,
            "green": None
        }

        # 按垂直位置排序（假设红在上，绿在下）
        results_sorted = sorted(
            results,
            key=lambda r: r.position[1] if r.position else 0
        )

        for result in results_sorted:
            color_key = result.color.value
            if color_key in andon and andon[color_key] is None:
                andon[color_key] = result

        # 填充未检测到的颜色
        for color in ["red", "yellow", "green"]:
            if andon[color] is None:
                andon[color] = LightResult(
                    color=LightColor(color),
                    state=LightState.OFF
                )

        return andon

    def _extract_roi(self, image: np.ndarray) -> np.ndarray:
        """提取感兴趣区域"""
        if self.config.region:
            x, y, w, h = self.config.region
            return image[y:y+h, x:x+w]
        return image

    def _detect_color(
        self,
        image: np.ndarray
    ) -> Tuple[LightColor, float, np.ndarray]:
        """
        检测主要颜色

        Returns:
            (颜色, 亮度, 掩码)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        best_color = LightColor.UNKNOWN
        best_score = 0
        best_mask = np.zeros(gray.shape, dtype=np.uint8)

        # 检测每种颜色
        for color, ranges in DEFAULT_COLOR_RANGES.items():
            mask = self._create_color_mask(hsv, ranges)

            # 计算该颜色的得分（非零像素比例 * 平均亮度）
            non_zero = cv2.countNonZero(mask)
            if non_zero > 0:
                brightness = cv2.mean(gray, mask=mask)[0]
                score = non_zero * brightness / (image.shape[0] * image.shape[1])

                if score > best_score:
                    best_score = score
                    best_color = color
                    best_mask = mask

        # 计算整体亮度
        overall_brightness = np.mean(gray)

        return best_color, overall_brightness, best_mask

    def _create_color_mask(
        self,
        hsv: np.ndarray,
        ranges: List[ColorRange]
    ) -> np.ndarray:
        """创建颜色掩码"""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color_range in ranges:
            lower = np.array(color_range.lower)
            upper = np.array(color_range.upper)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        return mask

    def _find_light_position(
        self,
        mask: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], int]:
        """
        找到指示灯位置

        Returns:
            (中心位置, 面积)
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, 0

        # 找最大轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # 计算中心
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy), int(area)

        return None, int(area)

    def _detect_blinking(self, is_on: bool) -> Tuple[LightState, float]:
        """
        检测闪烁状态

        Returns:
            (状态, 闪烁频率)
        """
        current_time = time.time()

        # 检测状态变化
        if self._last_state is not None:
            last_on = self._last_state == LightState.ON
            if is_on != last_on:
                self._state_changes += 1
                self._last_change_time = current_time

        self._last_state = LightState.ON if is_on else LightState.OFF

        # 判断是否闪烁
        if len(self._history) >= self.config.blink_window:
            changes = 0
            prev = self._history[0]
            for curr in list(self._history)[1:]:
                if curr != prev:
                    changes += 1
                prev = curr

            if changes >= self.config.blink_threshold:
                # 估算闪烁频率
                # 假设 30fps，窗口30帧 = 1秒
                freq = changes / 2.0  # 一个周期包含2次状态变化
                return LightState.BLINKING, freq

        if is_on:
            return LightState.ON, 0.0
        else:
            return LightState.OFF, 0.0

    def _calculate_confidence(
        self,
        brightness: float,
        area: int,
        mask: np.ndarray
    ) -> float:
        """计算识别置信度"""
        # 基于亮度的置信度
        brightness_conf = min(1.0, brightness / 200)

        # 基于面积的置信度
        if area < self.config.min_area:
            area_conf = 0.0
        elif area > self.config.max_area:
            area_conf = 0.5
        else:
            area_conf = min(1.0, area / 500)

        # 综合置信度
        return (brightness_conf + area_conf) / 2

    def reset(self):
        """重置状态"""
        self._history.clear()
        self._last_state = None
        self._state_changes = 0

    def visualize(self, image: np.ndarray, results: List[LightResult]) -> np.ndarray:
        """
        可视化识别结果

        Args:
            image: 原始图像
            results: 识别结果列表

        Returns:
            标注后的图像
        """
        output = image.copy()

        color_map = {
            LightColor.RED: (0, 0, 255),
            LightColor.YELLOW: (0, 255, 255),
            LightColor.GREEN: (0, 255, 0),
            LightColor.BLUE: (255, 0, 0),
            LightColor.WHITE: (255, 255, 255),
            LightColor.ORANGE: (0, 165, 255),
            LightColor.UNKNOWN: (128, 128, 128)
        }

        for result in results:
            if result.position:
                color = color_map.get(result.color, (128, 128, 128))

                # 绘制圆圈
                radius = max(10, int(np.sqrt(result.area / np.pi)))
                cv2.circle(output, result.position, radius, color, 2)

                # 绘制状态文字
                state_text = f"{result.color.value}: {result.state.value}"
                cv2.putText(
                    output, state_text,
                    (result.position[0] - 30, result.position[1] - radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        return output


class AndonMonitor:
    """
    Andon 灯监控器

    持续监控 Andon 灯状态，计算 OEE 相关指标

    使用示例:
    ```python
    monitor = AndonMonitor()

    while True:
        frame = camera.read()
        status = monitor.update(frame)

        print(f"运行时间: {status['runtime']}")
        print(f"停机时间: {status['downtime']}")
        print(f"可用率: {status['availability']:.2%}")
    ```
    """

    def __init__(self, config: Optional[LightConfig] = None):
        """初始化监控器"""
        self._recognizer = LightRecognizer(config)
        self._start_time = time.time()
        self._running_time = 0.0
        self._idle_time = 0.0
        self._fault_time = 0.0
        self._last_update = time.time()
        self._current_state = "unknown"

    def update(self, image: np.ndarray) -> Dict[str, Any]:
        """
        更新监控状态

        Args:
            image: 当前帧

        Returns:
            状态信息字典
        """
        current_time = time.time()
        dt = current_time - self._last_update
        self._last_update = current_time

        # 检测 Andon 灯
        andon = self._recognizer.detect_andon(image)

        # 判断设备状态
        red_on = andon["red"].state == LightState.ON
        yellow_on = andon["yellow"].state == LightState.ON
        green_on = andon["green"].state == LightState.ON

        if red_on:
            self._current_state = "fault"
            self._fault_time += dt
        elif yellow_on:
            self._current_state = "idle"
            self._idle_time += dt
        elif green_on:
            self._current_state = "running"
            self._running_time += dt
        else:
            self._current_state = "off"

        # 计算统计
        total_time = current_time - self._start_time
        availability = self._running_time / total_time if total_time > 0 else 0

        return {
            "state": self._current_state,
            "andon": {
                "red": andon["red"].state.value,
                "yellow": andon["yellow"].state.value,
                "green": andon["green"].state.value
            },
            "runtime": self._running_time,
            "idle_time": self._idle_time,
            "fault_time": self._fault_time,
            "total_time": total_time,
            "availability": availability
        }

    def reset(self):
        """重置统计"""
        self._start_time = time.time()
        self._running_time = 0.0
        self._idle_time = 0.0
        self._fault_time = 0.0
        self._last_update = time.time()


def detect_light(
    image: np.ndarray,
    region: Optional[Tuple[int, int, int, int]] = None
) -> LightResult:
    """
    便捷函数：检测指示灯

    Args:
        image: 输入图像
        region: 检测区域 (x, y, w, h)

    Returns:
        识别结果
    """
    config = LightConfig(region=region)
    recognizer = LightRecognizer(config)
    return recognizer.detect(image)


def detect_andon(image: np.ndarray) -> Dict[str, str]:
    """
    便捷函数：检测 Andon 灯状态

    Args:
        image: 输入图像

    Returns:
        {"red": "on/off", "yellow": "on/off", "green": "on/off"}
    """
    recognizer = LightRecognizer()
    andon = recognizer.detect_andon(image)
    return {
        color: result.state.value
        for color, result in andon.items()
    }

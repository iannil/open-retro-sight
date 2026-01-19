"""
开关/旋钮位置识别模块

功能：
- 拨动开关识别：ON/OFF 状态判断
- 旋钮档位检测：多档位旋钮位置
- 按钮状态：按下/弹起检测
- 刻度盘位置：角度到档位映射

用于设备状态监控和操作记录
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SwitchType(Enum):
    """开关类型"""
    TOGGLE = "toggle"       # 拨动开关
    ROTARY = "rotary"       # 旋转开关/旋钮
    PUSH_BUTTON = "button"  # 按钮
    SLIDER = "slider"       # 滑动开关
    SELECTOR = "selector"   # 选择器开关


class SwitchState(Enum):
    """开关状态"""
    ON = "on"
    OFF = "off"
    MIDDLE = "middle"     # 中间位置（三档开关）
    UNKNOWN = "unknown"


@dataclass
class SwitchConfig:
    """开关配置"""
    # 开关类型
    switch_type: SwitchType = SwitchType.TOGGLE
    # 检测区域 (x, y, width, height)
    region: Optional[Tuple[int, int, int, int]] = None
    # ON 状态参考图像路径
    on_reference: Optional[str] = None
    # OFF 状态参考图像路径
    off_reference: Optional[str] = None
    # 档位数量（旋钮）
    num_positions: int = 2
    # 档位角度列表（旋钮，度）
    position_angles: List[float] = field(default_factory=list)
    # 档位标签
    position_labels: List[str] = field(default_factory=list)
    # 匹配阈值 (0-1)
    match_threshold: float = 0.7
    # 颜色检测模式
    use_color: bool = False
    # ON 状态颜色范围 (HSV)
    on_color_lower: Tuple[int, int, int] = (0, 0, 0)
    on_color_upper: Tuple[int, int, int] = (180, 255, 255)


@dataclass
class SwitchResult:
    """开关识别结果"""
    state: SwitchState              # 状态
    position: int = 0               # 档位索引 (0-based)
    position_label: str = ""        # 档位标签
    confidence: float = 0.0         # 置信度
    angle: float = 0.0              # 角度（旋钮）
    center: Optional[Tuple[int, int]] = None  # 开关中心位置


class SwitchRecognizer:
    """
    开关/旋钮识别器

    使用示例:
    ```python
    # 拨动开关识别
    config = SwitchConfig(switch_type=SwitchType.TOGGLE)
    recognizer = SwitchRecognizer(config)
    result = recognizer.recognize(image)
    print(f"开关状态: {result.state}")

    # 旋钮档位识别
    config = SwitchConfig(
        switch_type=SwitchType.ROTARY,
        num_positions=4,
        position_labels=["OFF", "LOW", "MED", "HIGH"]
    )
    recognizer = SwitchRecognizer(config)
    result = recognizer.recognize(image)
    print(f"档位: {result.position_label}")
    ```
    """

    def __init__(self, config: Optional[SwitchConfig] = None):
        """
        初始化开关识别器

        Args:
            config: 开关配置
        """
        self.config = config or SwitchConfig()
        self._on_template: Optional[np.ndarray] = None
        self._off_template: Optional[np.ndarray] = None
        self._load_templates()

    def _load_templates(self):
        """加载参考模板图像"""
        if self.config.on_reference:
            self._on_template = cv2.imread(self.config.on_reference, cv2.IMREAD_GRAYSCALE)
        if self.config.off_reference:
            self._off_template = cv2.imread(self.config.off_reference, cv2.IMREAD_GRAYSCALE)

    def recognize(self, image: np.ndarray) -> SwitchResult:
        """
        识别开关状态

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            开关识别结果
        """
        # 提取 ROI
        roi = self._extract_roi(image)

        # 根据开关类型选择识别方法
        if self.config.switch_type == SwitchType.TOGGLE:
            return self._recognize_toggle(roi)
        elif self.config.switch_type == SwitchType.ROTARY:
            return self._recognize_rotary(roi)
        elif self.config.switch_type == SwitchType.PUSH_BUTTON:
            return self._recognize_button(roi)
        elif self.config.switch_type == SwitchType.SLIDER:
            return self._recognize_slider(roi)
        elif self.config.switch_type == SwitchType.SELECTOR:
            return self._recognize_selector(roi)
        else:
            return SwitchResult(state=SwitchState.UNKNOWN)

    def _extract_roi(self, image: np.ndarray) -> np.ndarray:
        """提取感兴趣区域"""
        if self.config.region:
            x, y, w, h = self.config.region
            return image[y:y+h, x:x+w]
        return image

    def _recognize_toggle(self, image: np.ndarray) -> SwitchResult:
        """
        识别拨动开关

        使用模板匹配或颜色检测判断 ON/OFF
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 方法1: 模板匹配
        if self._on_template is not None and self._off_template is not None:
            return self._match_templates(gray)

        # 方法2: 颜色检测
        if self.config.use_color:
            return self._detect_by_color(image)

        # 方法3: 位置检测（假设开关有明显的移动特征）
        return self._detect_toggle_position(gray)

    def _match_templates(self, gray: np.ndarray) -> SwitchResult:
        """使用模板匹配识别"""
        # 调整模板大小以匹配 ROI
        on_template = cv2.resize(self._on_template, (gray.shape[1], gray.shape[0]))
        off_template = cv2.resize(self._off_template, (gray.shape[1], gray.shape[0]))

        # 匹配
        on_result = cv2.matchTemplate(gray, on_template, cv2.TM_CCOEFF_NORMED)
        off_result = cv2.matchTemplate(gray, off_template, cv2.TM_CCOEFF_NORMED)

        on_score = np.max(on_result)
        off_score = np.max(off_result)

        if on_score > off_score and on_score > self.config.match_threshold:
            return SwitchResult(
                state=SwitchState.ON,
                position=1,
                position_label="ON",
                confidence=float(on_score)
            )
        elif off_score > self.config.match_threshold:
            return SwitchResult(
                state=SwitchState.OFF,
                position=0,
                position_label="OFF",
                confidence=float(off_score)
            )
        else:
            return SwitchResult(
                state=SwitchState.UNKNOWN,
                confidence=max(on_score, off_score)
            )

    def _detect_by_color(self, image: np.ndarray) -> SwitchResult:
        """使用颜色检测识别开关状态"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 检测 ON 状态颜色
        lower = np.array(self.config.on_color_lower)
        upper = np.array(self.config.on_color_upper)
        mask = cv2.inRange(hsv, lower, upper)

        # 计算颜色区域比例
        ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])

        if ratio > 0.3:  # 超过 30% 认为是 ON
            return SwitchResult(
                state=SwitchState.ON,
                position=1,
                position_label="ON",
                confidence=min(1.0, ratio * 2)
            )
        else:
            return SwitchResult(
                state=SwitchState.OFF,
                position=0,
                position_label="OFF",
                confidence=min(1.0, (1 - ratio) * 2)
            )

    def _detect_toggle_position(self, gray: np.ndarray) -> SwitchResult:
        """
        通过分析图像上下半区的亮度差异检测开关位置

        假设开关柄在 ON 时偏上，OFF 时偏下
        """
        h, w = gray.shape
        top_half = gray[:h//2, :]
        bottom_half = gray[h//2:, :]

        top_mean = np.mean(top_half)
        bottom_mean = np.mean(bottom_half)

        # 计算亮度差异
        diff = top_mean - bottom_mean
        total = top_mean + bottom_mean

        if total > 0:
            confidence = abs(diff) / total
        else:
            confidence = 0

        if diff > 10:  # 上半部分更亮 -> ON
            return SwitchResult(
                state=SwitchState.ON,
                position=1,
                position_label="ON",
                confidence=float(confidence)
            )
        elif diff < -10:  # 下半部分更亮 -> OFF
            return SwitchResult(
                state=SwitchState.OFF,
                position=0,
                position_label="OFF",
                confidence=float(confidence)
            )
        else:
            return SwitchResult(
                state=SwitchState.UNKNOWN,
                confidence=float(confidence)
            )

    def _recognize_rotary(self, image: np.ndarray) -> SwitchResult:
        """
        识别旋转开关/旋钮

        通过检测旋钮指针/标记的角度确定档位
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 霍夫线检测找到指针
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return SwitchResult(state=SwitchState.UNKNOWN)

        # 找到中心点
        h, w = gray.shape
        center = (w // 2, h // 2)

        # 找最长的线作为指针
        best_line = None
        max_length = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > max_length:
                max_length = length
                best_line = (x1, y1, x2, y2)

        if best_line is None:
            return SwitchResult(state=SwitchState.UNKNOWN)

        x1, y1, x2, y2 = best_line

        # 计算指针角度（相对于中心）
        # 选择离中心较远的端点
        dist1 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
        dist2 = np.sqrt((x2 - center[0])**2 + (y2 - center[1])**2)

        if dist1 > dist2:
            pointer_end = (x1, y1)
        else:
            pointer_end = (x2, y2)

        # 计算角度 (0度在正上方，顺时针增加)
        dx = pointer_end[0] - center[0]
        dy = center[1] - pointer_end[1]  # 注意 y 轴反向
        angle = np.degrees(np.arctan2(dx, dy))
        if angle < 0:
            angle += 360

        # 根据角度确定档位
        position, label, confidence = self._angle_to_position(angle)

        return SwitchResult(
            state=SwitchState.ON if position > 0 else SwitchState.OFF,
            position=position,
            position_label=label,
            confidence=confidence,
            angle=angle,
            center=center
        )

    def _angle_to_position(self, angle: float) -> Tuple[int, str, float]:
        """
        将角度映射到档位

        Returns:
            (档位索引, 档位标签, 置信度)
        """
        if self.config.position_angles:
            # 使用配置的角度
            angles = self.config.position_angles
        else:
            # 均匀分布
            n = self.config.num_positions
            angles = [i * 360 / n for i in range(n)]

        # 找到最近的档位
        min_diff = float('inf')
        best_pos = 0

        for i, pos_angle in enumerate(angles):
            # 计算角度差（考虑环绕）
            diff = abs(angle - pos_angle)
            if diff > 180:
                diff = 360 - diff

            if diff < min_diff:
                min_diff = diff
                best_pos = i

        # 计算置信度（角度越接近越高）
        max_diff = 180 / self.config.num_positions
        confidence = max(0, 1 - min_diff / max_diff)

        # 获取标签
        if self.config.position_labels and best_pos < len(self.config.position_labels):
            label = self.config.position_labels[best_pos]
        else:
            label = f"Position {best_pos}"

        return best_pos, label, confidence

    def _recognize_button(self, image: np.ndarray) -> SwitchResult:
        """
        识别按钮状态

        通过亮度/阴影变化判断按下/弹起
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算边缘强度（按下时阴影变化）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])

        # 计算中心区域亮度
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_brightness = np.mean(center_region)

        # 计算边缘区域亮度
        edge_brightness = (np.sum(gray) - np.sum(center_region)) / (h*w - center_region.size)

        # 按下时中心通常更暗，边缘更亮（反光效果）
        brightness_diff = edge_brightness - center_brightness

        if brightness_diff > 10 and edge_density > 0.05:
            return SwitchResult(
                state=SwitchState.ON,  # 按下
                position=1,
                position_label="PRESSED",
                confidence=min(1.0, brightness_diff / 30)
            )
        else:
            return SwitchResult(
                state=SwitchState.OFF,  # 弹起
                position=0,
                position_label="RELEASED",
                confidence=min(1.0, 1 - brightness_diff / 30)
            )

    def _recognize_slider(self, image: np.ndarray) -> SwitchResult:
        """
        识别滑动开关

        检测滑块位置
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 假设滑动开关是水平的
        h, w = gray.shape

        # 使用阈值找到滑块
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算每列的白色像素数
        col_sums = np.sum(binary, axis=0)

        # 找到滑块位置（白色像素最集中的位置）
        if np.max(col_sums) > 0:
            slider_pos = np.argmax(col_sums)
            position_ratio = slider_pos / w

            # 根据位置确定状态
            if position_ratio < 0.4:
                state = SwitchState.OFF
                position = 0
                label = "OFF"
            elif position_ratio > 0.6:
                state = SwitchState.ON
                position = 1
                label = "ON"
            else:
                state = SwitchState.MIDDLE
                position = -1
                label = "MIDDLE"

            confidence = abs(position_ratio - 0.5) * 2

            return SwitchResult(
                state=state,
                position=position,
                position_label=label,
                confidence=float(confidence),
                center=(slider_pos, h // 2)
            )

        return SwitchResult(state=SwitchState.UNKNOWN)

    def _recognize_selector(self, image: np.ndarray) -> SwitchResult:
        """
        识别选择器开关

        类似旋钮但可能有更多离散位置
        """
        # 复用旋钮识别逻辑
        return self._recognize_rotary(image)

    def set_templates(
        self,
        on_image: np.ndarray,
        off_image: np.ndarray
    ):
        """
        设置参考模板

        Args:
            on_image: ON 状态参考图像
            off_image: OFF 状态参考图像
        """
        self._on_template = cv2.cvtColor(on_image, cv2.COLOR_BGR2GRAY)
        self._off_template = cv2.cvtColor(off_image, cv2.COLOR_BGR2GRAY)

    def visualize(
        self,
        image: np.ndarray,
        result: SwitchResult
    ) -> np.ndarray:
        """
        可视化识别结果

        Args:
            image: 原始图像
            result: 识别结果

        Returns:
            标注后的图像
        """
        output = image.copy()

        # 绘制状态文字
        text = f"{result.position_label} ({result.confidence:.2f})"
        color = (0, 255, 0) if result.state == SwitchState.ON else (0, 0, 255)

        cv2.putText(
            output, text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        # 绘制中心点和指针（旋钮）
        if result.center and self.config.switch_type == SwitchType.ROTARY:
            cv2.circle(output, result.center, 5, (255, 0, 0), -1)

            # 绘制指针方向
            angle_rad = np.radians(result.angle - 90)  # 转换坐标系
            end_x = int(result.center[0] + 50 * np.cos(angle_rad))
            end_y = int(result.center[1] + 50 * np.sin(angle_rad))
            cv2.line(output, result.center, (end_x, end_y), (0, 255, 255), 2)

        return output


class MultiSwitchMonitor:
    """
    多开关监控器

    同时监控多个开关状态

    使用示例:
    ```python
    monitor = MultiSwitchMonitor()
    monitor.add_switch("power", SwitchConfig(region=(100, 100, 50, 50)))
    monitor.add_switch("mode", SwitchConfig(
        switch_type=SwitchType.ROTARY,
        region=(200, 100, 80, 80),
        num_positions=3,
        position_labels=["OFF", "AUTO", "MANUAL"]
    ))

    results = monitor.update(frame)
    for name, result in results.items():
        print(f"{name}: {result.state}")
    ```
    """

    def __init__(self):
        """初始化多开关监控器"""
        self._switches: Dict[str, SwitchRecognizer] = {}
        self._last_results: Dict[str, SwitchResult] = {}

    def add_switch(self, name: str, config: SwitchConfig):
        """
        添加开关

        Args:
            name: 开关名称
            config: 开关配置
        """
        self._switches[name] = SwitchRecognizer(config)
        logger.info(f"Added switch: {name}")

    def remove_switch(self, name: str):
        """移除开关"""
        if name in self._switches:
            del self._switches[name]
            if name in self._last_results:
                del self._last_results[name]

    def update(self, image: np.ndarray) -> Dict[str, SwitchResult]:
        """
        更新所有开关状态

        Args:
            image: 输入图像

        Returns:
            {开关名称: 识别结果}
        """
        results = {}

        for name, recognizer in self._switches.items():
            try:
                result = recognizer.recognize(image)
                results[name] = result
                self._last_results[name] = result
            except Exception as e:
                logger.error(f"Error recognizing switch {name}: {e}")
                # 保持上一次结果
                if name in self._last_results:
                    results[name] = self._last_results[name]
                else:
                    results[name] = SwitchResult(state=SwitchState.UNKNOWN)

        return results

    def get_state(self, name: str) -> Optional[SwitchResult]:
        """获取开关状态"""
        return self._last_results.get(name)

    def get_all_states(self) -> Dict[str, SwitchResult]:
        """获取所有开关状态"""
        return self._last_results.copy()

    @property
    def switch_names(self) -> List[str]:
        """获取所有开关名称"""
        return list(self._switches.keys())


def detect_switch(
    image: np.ndarray,
    region: Optional[Tuple[int, int, int, int]] = None,
    switch_type: SwitchType = SwitchType.TOGGLE
) -> SwitchResult:
    """
    便捷函数：检测开关状态

    Args:
        image: 输入图像
        region: 检测区域 (x, y, w, h)
        switch_type: 开关类型

    Returns:
        识别结果
    """
    config = SwitchConfig(region=region, switch_type=switch_type)
    recognizer = SwitchRecognizer(config)
    return recognizer.recognize(image)


def detect_rotary(
    image: np.ndarray,
    num_positions: int = 4,
    position_labels: Optional[List[str]] = None,
    region: Optional[Tuple[int, int, int, int]] = None
) -> SwitchResult:
    """
    便捷函数：检测旋钮档位

    Args:
        image: 输入图像
        num_positions: 档位数量
        position_labels: 档位标签
        region: 检测区域

    Returns:
        识别结果
    """
    config = SwitchConfig(
        switch_type=SwitchType.ROTARY,
        region=region,
        num_positions=num_positions,
        position_labels=position_labels or []
    )
    recognizer = SwitchRecognizer(config)
    return recognizer.recognize(image)

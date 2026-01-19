"""
指针式仪表识别模块

功能：
- 指针角度检测：基于 Hough 变换
- 刻度盘识别：自动检测刻度范围
- 角度到数值映射：线性/非线性插值
- 多指针支持：时分秒针、双指针仪表

用于读取压力表、温度计、速度表等模拟仪表
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


class GaugeType(Enum):
    """仪表类型"""
    CIRCULAR = "circular"       # 圆形表盘
    SEMICIRCLE = "semicircle"   # 半圆表盘
    ARC = "arc"                 # 扇形表盘
    LINEAR = "linear"           # 线性表盘


@dataclass
class GaugeConfig:
    """仪表配置"""
    gauge_type: GaugeType = GaugeType.CIRCULAR
    center: Optional[Tuple[int, int]] = None     # 表盘中心点
    radius: Optional[int] = None                  # 表盘半径
    min_angle: float = 225.0      # 最小值对应角度（度，12点钟为0，顺时针）
    max_angle: float = -45.0      # 最大值对应角度
    min_value: float = 0.0        # 最小刻度值
    max_value: float = 100.0      # 最大刻度值
    unit: str = ""                # 单位
    pointer_color: str = "dark"   # 指针颜色 ("dark", "light", "red")


@dataclass
class PointerResult:
    """指针识别结果"""
    angle: float                  # 检测到的角度（度）
    value: float                  # 映射后的数值
    confidence: float = 0.0       # 置信度
    center: Optional[Tuple[int, int]] = None  # 检测到的中心点
    tip: Optional[Tuple[int, int]] = None     # 指针尖端位置
    raw_lines: List = field(default_factory=list)  # 原始检测到的线段


@dataclass
class CalibrationPoint:
    """校准点"""
    angle: float      # 角度（度）
    value: float      # 对应的实际值


@dataclass
class CalibrationData:
    """校准数据"""
    points: List[CalibrationPoint] = field(default_factory=list)
    method: str = "linear"  # "linear" 或 "polynomial"
    coefficients: Optional[List[float]] = None  # 多项式系数（用于非线性校准）

    def is_valid(self) -> bool:
        """检查校准数据是否有效"""
        return len(self.points) >= 2

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "points": [{"angle": p.angle, "value": p.value} for p in self.points],
            "method": self.method,
            "coefficients": self.coefficients
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationData":
        """从字典创建"""
        points = [CalibrationPoint(**p) for p in data.get("points", [])]
        return cls(
            points=points,
            method=data.get("method", "linear"),
            coefficients=data.get("coefficients")
        )


class PointerRecognizer:
    """
    指针识别器

    使用示例:
    ```python
    # 配置仪表参数
    config = GaugeConfig(
        center=(200, 200),
        radius=150,
        min_angle=225,
        max_angle=-45,
        min_value=0,
        max_value=100,
        unit="MPa"
    )

    recognizer = PointerRecognizer(config)
    result = recognizer.recognize(image)

    print(f"读数: {result.value} {config.unit}")
    ```
    """

    def __init__(self, config: Optional[GaugeConfig] = None):
        """
        初始化指针识别器

        Args:
            config: 仪表配置
        """
        self.config = config or GaugeConfig()
        self.calibration: Optional[CalibrationData] = None

    def recognize(self, image: np.ndarray) -> PointerResult:
        """
        识别图像中的指针读数

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            指针识别结果
        """
        # 自动检测中心和半径
        if self.config.center is None or self.config.radius is None:
            center, radius = self._detect_dial(image)
            if center is not None:
                self.config.center = center
                self.config.radius = radius

        if self.config.center is None:
            return PointerResult(angle=0, value=0, confidence=0)

        # 预处理图像
        processed = self._preprocess(image)

        # 检测指针
        angle, confidence, tip, lines = self._detect_pointer(processed)

        # 角度映射到数值
        value = self._angle_to_value(angle)

        return PointerResult(
            angle=angle,
            value=value,
            confidence=confidence,
            center=self.config.center,
            tip=tip,
            raw_lines=lines
        )

    def recognize_multi(self, image: np.ndarray, num_pointers: int = 2) -> List[PointerResult]:
        """
        识别多个指针（如时钟的时分秒针）

        Args:
            image: 输入图像
            num_pointers: 预期指针数量

        Returns:
            指针识别结果列表
        """
        if self.config.center is None or self.config.radius is None:
            center, radius = self._detect_dial(image)
            if center is not None:
                self.config.center = center
                self.config.radius = radius

        if self.config.center is None:
            return []

        processed = self._preprocess(image)
        lines = self._detect_lines(processed)

        # 按长度分组线段
        pointer_lines = self._group_lines_by_length(lines, num_pointers)

        results = []
        for group in pointer_lines:
            angle, tip = self._calculate_angle_from_lines(group)
            value = self._angle_to_value(angle)
            results.append(PointerResult(
                angle=angle,
                value=value,
                confidence=0.8,
                center=self.config.center,
                tip=tip,
                raw_lines=group
            ))

        return results

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理

        针对指针检测优化
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 根据指针颜色选择处理方式
        if self.config.pointer_color == "dark":
            # 深色指针：反转后检测
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif self.config.pointer_color == "red":
            # 红色指针：提取红色通道
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # 红色范围
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])
                upper_red2 = np.array([180, 255, 255])
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                binary = cv2.bitwise_or(mask1, mask2)
            else:
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # 浅色指针
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def _detect_dial(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """
        自动检测表盘中心和半径

        Returns:
            (中心点, 半径)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # 霍夫圆检测
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=30,
            maxRadius=min(image.shape[:2]) // 2
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 取最大的圆
            largest = max(circles[0], key=lambda c: c[2])
            center = (int(largest[0]), int(largest[1]))
            radius = int(largest[2])
            return center, radius

        # 如果检测失败，使用图像中心
        h, w = image.shape[:2]
        return (w // 2, h // 2), min(w, h) // 3

    def _detect_lines(self, binary: np.ndarray) -> List:
        """检测线段"""
        # 边缘检测
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        # 霍夫线变换
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10
        )

        return lines if lines is not None else []

    def _detect_pointer(self, binary: np.ndarray) -> Tuple[float, float, Optional[Tuple[int, int]], List]:
        """
        检测指针角度

        Returns:
            (角度, 置信度, 指针尖端, 线段列表)
        """
        lines = self._detect_lines(binary)

        if len(lines) == 0:
            return 0.0, 0.0, None, []

        center = self.config.center
        radius = self.config.radius

        # 筛选经过中心附近的线段
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算线段到中心的距离
            dist = self._point_line_distance(center, (x1, y1), (x2, y2))

            if dist < radius * 0.3:  # 距中心30%半径内
                valid_lines.append(line[0])

        if not valid_lines:
            return 0.0, 0.0, None, []

        # 计算指针角度
        angle, tip = self._calculate_angle_from_lines(valid_lines)

        # 计算置信度（基于检测到的线段数量和一致性）
        confidence = min(1.0, len(valid_lines) / 5.0)

        return angle, confidence, tip, valid_lines

    def _calculate_angle_from_lines(
        self,
        lines: List
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        从线段计算指针角度

        Returns:
            (角度, 指针尖端坐标)
        """
        if not lines:
            return 0.0, None

        center = self.config.center
        angles = []
        tips = []

        for line in lines:
            if isinstance(line, np.ndarray):
                x1, y1, x2, y2 = line
            else:
                x1, y1, x2, y2 = line

            # 确定哪个端点是指针尖端（距离中心更远的）
            dist1 = math.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
            dist2 = math.sqrt((x2 - center[0])**2 + (y2 - center[1])**2)

            if dist1 > dist2:
                tip_x, tip_y = x1, y1
            else:
                tip_x, tip_y = x2, y2

            # 计算角度（相对于中心，12点钟方向为0度，顺时针为正）
            dx = tip_x - center[0]
            dy = tip_y - center[1]

            # atan2 返回 -π 到 π，转换为 0-360 度
            angle = math.degrees(math.atan2(dx, -dy))  # 注意 y 轴反转
            if angle < 0:
                angle += 360

            angles.append(angle)
            tips.append((tip_x, tip_y))

        # 使用中位数角度
        median_angle = np.median(angles)

        # 找到最接近中位数角度的尖端
        best_idx = min(range(len(angles)), key=lambda i: abs(angles[i] - median_angle))
        tip = tips[best_idx]

        return median_angle, tip

    def _angle_to_value(self, angle: float) -> float:
        """
        将角度映射到刻度值

        Args:
            angle: 检测到的角度（0-360度）

        Returns:
            对应的刻度值
        """
        # 如果有校准数据，使用校准后的映射
        if self.calibration is not None and self.calibration.is_valid():
            return self._calibrated_angle_to_value(angle)

        # 默认：使用配置中的线性映射
        min_angle = self.config.min_angle
        max_angle = self.config.max_angle

        # 处理角度跨越 0/360 度的情况
        if min_angle > max_angle:
            # 例如 min=225, max=-45 (即315度范围)
            if angle > min_angle:
                angle_range = 360 - min_angle + max_angle + 360
                current_pos = angle - min_angle
            else:
                angle_range = 360 - min_angle + max_angle + 360
                current_pos = 360 - min_angle + angle
        else:
            angle_range = max_angle - min_angle
            current_pos = angle - min_angle

        # 线性插值
        ratio = current_pos / angle_range if angle_range != 0 else 0
        ratio = max(0, min(1, ratio))  # 限制在 0-1 范围

        value = self.config.min_value + ratio * (self.config.max_value - self.config.min_value)
        return value

    def _calibrated_angle_to_value(self, angle: float) -> float:
        """
        使用校准数据将角度映射到数值

        Args:
            angle: 检测到的角度

        Returns:
            校准后的数值
        """
        if self.calibration is None or self.calibration.coefficients is None:
            return 0.0

        coeffs = self.calibration.coefficients

        if self.calibration.method == "linear":
            # value = a * angle + b
            a, b = coeffs[0], coeffs[1]
            return a * angle + b

        elif self.calibration.method == "polynomial":
            # value = a * angle^2 + b * angle + c (使用 numpy polyval)
            return float(np.polyval(coeffs, angle))

        return 0.0

    def _point_line_distance(
        self,
        point: Tuple[int, int],
        line_p1: Tuple[int, int],
        line_p2: Tuple[int, int]
    ) -> float:
        """计算点到线段的距离"""
        px, py = point
        x1, y1 = line_p1
        x2, y2 = line_p2

        # 线段长度
        line_len = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_len == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        # 点到直线的距离
        dist = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_len
        return dist

    def _group_lines_by_length(self, lines: List, num_groups: int) -> List[List]:
        """
        按长度将线段分组

        用于识别多个不同长度的指针
        """
        if not lines or len(lines) == 0:
            return []

        # 计算每条线段的长度
        line_lengths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            line_lengths.append((length, line[0]))

        # 按长度排序
        line_lengths.sort(key=lambda x: x[0], reverse=True)

        # 简单分组：按长度等分
        groups = [[] for _ in range(num_groups)]
        for i, (length, line) in enumerate(line_lengths):
            group_idx = min(i * num_groups // len(line_lengths), num_groups - 1)
            groups[group_idx].append(line)

        return [g for g in groups if g]

    def calibrate(
        self,
        image: np.ndarray,
        known_value: float,
        known_angle: Optional[float] = None
    ):
        """
        使用已知值添加校准点

        Args:
            image: 包含已知读数的图像
            known_value: 已知的正确读数
            known_angle: 已知的角度（可选，如果不提供则自动检测）
        """
        if known_angle is None:
            result = self.recognize(image)
            known_angle = result.angle

        # 初始化校准数据
        if self.calibration is None:
            self.calibration = CalibrationData()

        # 添加校准点
        point = CalibrationPoint(angle=known_angle, value=known_value)
        self.calibration.points.append(point)

        # 如果有足够的点，计算校准系数
        if len(self.calibration.points) >= 2:
            self._compute_calibration()

        logger.info(f"校准点已添加: 角度 {known_angle:.1f}° 对应值 {known_value}")

    def calibrate_two_point(
        self,
        angle1: float,
        value1: float,
        angle2: float,
        value2: float
    ):
        """
        两点校准（线性）

        Args:
            angle1: 第一个校准点的角度
            value1: 第一个校准点的实际值
            angle2: 第二个校准点的角度
            value2: 第二个校准点的实际值
        """
        self.calibration = CalibrationData(
            points=[
                CalibrationPoint(angle=angle1, value=value1),
                CalibrationPoint(angle=angle2, value=value2)
            ],
            method="linear"
        )
        self._compute_calibration()
        logger.info(f"两点校准完成: ({angle1:.1f}°, {value1}) → ({angle2:.1f}°, {value2})")

    def calibrate_three_point(
        self,
        angle1: float,
        value1: float,
        angle2: float,
        value2: float,
        angle3: float,
        value3: float
    ):
        """
        三点校准（二次多项式，用于非线性刻度）

        Args:
            angle1, value1: 第一个校准点
            angle2, value2: 第二个校准点（通常是中间点）
            angle3, value3: 第三个校准点
        """
        self.calibration = CalibrationData(
            points=[
                CalibrationPoint(angle=angle1, value=value1),
                CalibrationPoint(angle=angle2, value=value2),
                CalibrationPoint(angle=angle3, value=value3)
            ],
            method="polynomial"
        )
        self._compute_calibration()
        logger.info(f"三点校准完成: 非线性多项式拟合")

    def _compute_calibration(self):
        """计算校准系数"""
        if self.calibration is None or len(self.calibration.points) < 2:
            return

        points = self.calibration.points
        angles = [p.angle for p in points]
        values = [p.value for p in points]

        if len(points) == 2:
            # 线性校准: value = a * angle + b
            self.calibration.method = "linear"
            a = (values[1] - values[0]) / (angles[1] - angles[0]) if angles[1] != angles[0] else 0
            b = values[0] - a * angles[0]
            self.calibration.coefficients = [a, b]

        elif len(points) >= 3:
            # 多项式校准 (二次): value = a * angle^2 + b * angle + c
            self.calibration.method = "polynomial"
            try:
                # 使用最小二乘法拟合二次多项式
                coefficients = np.polyfit(angles, values, 2)
                self.calibration.coefficients = coefficients.tolist()
            except Exception as e:
                logger.warning(f"多项式拟合失败，回退到线性: {e}")
                self.calibration.method = "linear"
                a = (values[-1] - values[0]) / (angles[-1] - angles[0]) if angles[-1] != angles[0] else 0
                b = values[0] - a * angles[0]
                self.calibration.coefficients = [a, b]

    def clear_calibration(self):
        """清除校准数据"""
        self.calibration = None
        logger.info("校准数据已清除")

    def save_calibration(self, path: str):
        """
        保存校准数据到文件

        Args:
            path: 文件路径（JSON 格式）
        """
        import json

        if self.calibration is None:
            raise ValueError("没有校准数据可保存")

        data = {
            "calibration": self.calibration.to_dict(),
            "config": {
                "gauge_type": self.config.gauge_type.value,
                "center": self.config.center,
                "radius": self.config.radius,
                "min_angle": self.config.min_angle,
                "max_angle": self.config.max_angle,
                "min_value": self.config.min_value,
                "max_value": self.config.max_value,
                "unit": self.config.unit
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"校准数据已保存到: {path}")

    def load_calibration(self, path: str):
        """
        从文件加载校准数据

        Args:
            path: 文件路径（JSON 格式）
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 加载校准数据
        if "calibration" in data:
            self.calibration = CalibrationData.from_dict(data["calibration"])

        # 可选：加载配置
        if "config" in data:
            cfg = data["config"]
            if "gauge_type" in cfg:
                self.config.gauge_type = GaugeType(cfg["gauge_type"])
            if "center" in cfg and cfg["center"]:
                self.config.center = tuple(cfg["center"])
            if "radius" in cfg:
                self.config.radius = cfg["radius"]
            if "min_angle" in cfg:
                self.config.min_angle = cfg["min_angle"]
            if "max_angle" in cfg:
                self.config.max_angle = cfg["max_angle"]
            if "min_value" in cfg:
                self.config.min_value = cfg["min_value"]
            if "max_value" in cfg:
                self.config.max_value = cfg["max_value"]
            if "unit" in cfg:
                self.config.unit = cfg["unit"]

        logger.info(f"校准数据已从 {path} 加载")

    def visualize(self, image: np.ndarray, result: PointerResult) -> np.ndarray:
        """
        可视化识别结果

        Args:
            image: 原始图像
            result: 识别结果

        Returns:
            标注后的图像
        """
        output = image.copy()

        if self.config.center:
            # 绘制中心点
            cv2.circle(output, self.config.center, 5, (0, 255, 0), -1)

            # 绘制表盘轮廓
            if self.config.radius:
                cv2.circle(output, self.config.center, self.config.radius, (0, 255, 0), 2)

        # 绘制检测到的线段
        for line in result.raw_lines:
            if isinstance(line, np.ndarray):
                x1, y1, x2, y2 = line
            else:
                x1, y1, x2, y2 = line
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 绘制指针尖端
        if result.tip:
            cv2.circle(output, result.tip, 8, (0, 0, 255), -1)

            # 绘制从中心到尖端的线
            if self.config.center:
                cv2.line(output, self.config.center, result.tip, (0, 0, 255), 2)

        # 显示读数
        text = f"{result.value:.1f} {self.config.unit}"
        cv2.putText(
            output, text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        return output


def recognize_gauge(
    image: np.ndarray,
    min_value: float = 0,
    max_value: float = 100,
    unit: str = ""
) -> PointerResult:
    """
    便捷函数：识别仪表读数

    Args:
        image: 仪表图像
        min_value: 最小刻度值
        max_value: 最大刻度值
        unit: 单位

    Returns:
        识别结果
    """
    config = GaugeConfig(
        min_value=min_value,
        max_value=max_value,
        unit=unit
    )
    recognizer = PointerRecognizer(config)
    return recognizer.recognize(image)

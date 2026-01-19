"""
数据平滑与滤波模块

提供多种滤波算法：
- 卡尔曼滤波 (Kalman Filter)
- 滑动平均 (Moving Average)
- 指数平滑 (Exponential Smoothing)
- 异常值过滤 (Outlier Filter)

用于平滑OCR识别结果，过滤噪声和异常值
"""

import numpy as np
from typing import Optional, List, Deque
from collections import deque
from dataclasses import dataclass
import time


@dataclass
class FilterConfig:
    """滤波器配置"""
    window_size: int = 5              # 滑动窗口大小
    process_variance: float = 1e-5    # 卡尔曼滤波过程噪声
    measure_variance: float = 1e-1    # 卡尔曼滤波测量噪声
    alpha: float = 0.3                # 指数平滑系数
    outlier_threshold: float = 3.0    # 异常值阈值（标准差倍数）
    min_value: Optional[float] = None # 最小有效值
    max_value: Optional[float] = None # 最大有效值
    max_change_rate: Optional[float] = None  # 最大变化率


class KalmanFilter1D:
    """
    一维卡尔曼滤波器

    适用于平滑连续变化的测量值，如温度、压力等

    使用示例:
    ```python
    kf = KalmanFilter1D()
    smoothed = kf.update(measurement)
    ```
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measure_variance: float = 1e-1,
        initial_value: float = 0.0
    ):
        """
        初始化卡尔曼滤波器

        Args:
            process_variance: 过程噪声方差（越小越相信模型预测）
            measure_variance: 测量噪声方差（越小越相信测量值）
            initial_value: 初始估计值
        """
        self.process_variance = process_variance
        self.measure_variance = measure_variance
        self.estimate = initial_value
        self.error_estimate = 1.0
        self._initialized = False

    def update(self, measurement: float) -> float:
        """
        更新滤波器并返回平滑值

        Args:
            measurement: 测量值

        Returns:
            平滑后的估计值
        """
        if not self._initialized:
            self.estimate = measurement
            self._initialized = True
            return measurement

        # 预测步骤
        prediction = self.estimate
        prediction_error = self.error_estimate + self.process_variance

        # 更新步骤
        kalman_gain = prediction_error / (prediction_error + self.measure_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * prediction_error

        return self.estimate

    def reset(self, value: float = 0.0):
        """重置滤波器"""
        self.estimate = value
        self.error_estimate = 1.0
        self._initialized = False


class MovingAverage:
    """
    滑动平均滤波器

    简单有效的平滑方法，适用于稳定信号

    使用示例:
    ```python
    ma = MovingAverage(window_size=5)
    smoothed = ma.update(measurement)
    ```
    """

    def __init__(self, window_size: int = 5):
        """
        初始化滑动平均滤波器

        Args:
            window_size: 窗口大小
        """
        self.window_size = window_size
        self._window: Deque[float] = deque(maxlen=window_size)

    def update(self, measurement: float) -> float:
        """
        更新滤波器并返回平滑值

        Args:
            measurement: 测量值

        Returns:
            平滑后的值
        """
        self._window.append(measurement)
        return sum(self._window) / len(self._window)

    @property
    def value(self) -> float:
        """当前平均值"""
        return sum(self._window) / len(self._window) if self._window else 0.0

    def reset(self):
        """重置滤波器"""
        self._window.clear()


class ExponentialSmoothing:
    """
    指数平滑滤波器

    对近期数据赋予更高权重

    使用示例:
    ```python
    es = ExponentialSmoothing(alpha=0.3)
    smoothed = es.update(measurement)
    ```
    """

    def __init__(self, alpha: float = 0.3, initial_value: float = 0.0):
        """
        初始化指数平滑滤波器

        Args:
            alpha: 平滑系数 (0-1)，越大越响应快
            initial_value: 初始值
        """
        self.alpha = alpha
        self._value = initial_value
        self._initialized = False

    def update(self, measurement: float) -> float:
        """
        更新滤波器并返回平滑值

        Args:
            measurement: 测量值

        Returns:
            平滑后的值
        """
        if not self._initialized:
            self._value = measurement
            self._initialized = True
            return measurement

        self._value = self.alpha * measurement + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        """当前平滑值"""
        return self._value

    def reset(self, value: float = 0.0):
        """重置滤波器"""
        self._value = value
        self._initialized = False


class OutlierFilter:
    """
    异常值过滤器

    基于历史数据的统计特性过滤异常值

    使用示例:
    ```python
    of = OutlierFilter(threshold=3.0)
    filtered = of.filter(measurement)  # 返回 None 如果是异常值
    ```
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 3.0,
        min_samples: int = 3
    ):
        """
        初始化异常值过滤器

        Args:
            window_size: 历史窗口大小
            threshold: 异常值阈值（标准差倍数）
            min_samples: 开始过滤前的最小样本数
        """
        self.threshold = threshold
        self.min_samples = min_samples
        self._history: Deque[float] = deque(maxlen=window_size)

    def filter(self, measurement: float) -> Optional[float]:
        """
        过滤异常值

        Args:
            measurement: 测量值

        Returns:
            有效值或 None（如果是异常值）
        """
        if len(self._history) < self.min_samples:
            self._history.append(measurement)
            return measurement

        mean = np.mean(self._history)
        std = np.std(self._history)

        # 防止除零
        if std < 1e-10:
            std = 1e-10

        # 判断是否为异常值
        z_score = abs(measurement - mean) / std

        if z_score > self.threshold:
            return None  # 异常值

        self._history.append(measurement)
        return measurement

    def filter_or_last(self, measurement: float) -> float:
        """
        过滤异常值，如果是异常值则返回上一个有效值

        Args:
            measurement: 测量值

        Returns:
            有效值
        """
        result = self.filter(measurement)
        if result is None:
            return self._history[-1] if self._history else measurement
        return result

    def reset(self):
        """重置滤波器"""
        self._history.clear()


class ValueValidator:
    """
    数值校验器

    基于业务规则校验数值有效性

    使用示例:
    ```python
    validator = ValueValidator(min_value=0, max_value=100, max_change_rate=10)
    valid, value = validator.validate(measurement)
    ```
    """

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        max_change_rate: Optional[float] = None
    ):
        """
        初始化校验器

        Args:
            min_value: 最小有效值
            max_value: 最大有效值
            max_change_rate: 最大变化率（每秒）
        """
        self.min_value = min_value
        self.max_value = max_value
        self.max_change_rate = max_change_rate
        self._last_value: Optional[float] = None
        self._last_time: Optional[float] = None

    def validate(self, measurement: float) -> tuple:
        """
        校验数值

        Args:
            measurement: 测量值

        Returns:
            (是否有效, 校正后的值)
        """
        current_time = time.time()
        corrected_value = measurement

        # 范围检查
        if self.min_value is not None and measurement < self.min_value:
            return (False, self._last_value or self.min_value)

        if self.max_value is not None and measurement > self.max_value:
            return (False, self._last_value or self.max_value)

        # 变化率检查
        if (
            self.max_change_rate is not None
            and self._last_value is not None
            and self._last_time is not None
        ):
            dt = current_time - self._last_time
            if dt > 0:
                change_rate = abs(measurement - self._last_value) / dt
                if change_rate > self.max_change_rate:
                    return (False, self._last_value)

        # 更新历史
        self._last_value = measurement
        self._last_time = current_time

        return (True, corrected_value)

    def reset(self):
        """重置校验器"""
        self._last_value = None
        self._last_time = None


class CompositeFilter:
    """
    组合滤波器

    将多个滤波器串联使用

    使用示例:
    ```python
    cf = CompositeFilter()
    cf.add_outlier_filter(threshold=3.0)
    cf.add_kalman_filter()
    smoothed = cf.filter(measurement)
    ```
    """

    def __init__(self):
        """初始化组合滤波器"""
        self._filters: List = []
        self._validator: Optional[ValueValidator] = None

    def add_kalman_filter(
        self,
        process_variance: float = 1e-5,
        measure_variance: float = 1e-1
    ) -> "CompositeFilter":
        """添加卡尔曼滤波器"""
        self._filters.append(
            KalmanFilter1D(process_variance, measure_variance)
        )
        return self

    def add_moving_average(self, window_size: int = 5) -> "CompositeFilter":
        """添加滑动平均滤波器"""
        self._filters.append(MovingAverage(window_size))
        return self

    def add_exponential_smoothing(self, alpha: float = 0.3) -> "CompositeFilter":
        """添加指数平滑滤波器"""
        self._filters.append(ExponentialSmoothing(alpha))
        return self

    def add_outlier_filter(
        self,
        threshold: float = 3.0,
        window_size: int = 10
    ) -> "CompositeFilter":
        """添加异常值过滤器"""
        self._filters.append(OutlierFilter(window_size, threshold))
        return self

    def set_validator(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        max_change_rate: Optional[float] = None
    ) -> "CompositeFilter":
        """设置数值校验器"""
        self._validator = ValueValidator(min_value, max_value, max_change_rate)
        return self

    def filter(self, measurement: float) -> Optional[float]:
        """
        应用所有滤波器

        Args:
            measurement: 测量值

        Returns:
            滤波后的值，如果无效返回 None
        """
        value = measurement

        # 先校验
        if self._validator:
            valid, value = self._validator.validate(value)
            if not valid:
                return value

        # 依次通过滤波器
        for f in self._filters:
            if isinstance(f, OutlierFilter):
                result = f.filter(value)
                if result is None:
                    return f.filter_or_last(value)
                value = result
            else:
                value = f.update(value)

        return value

    def reset(self):
        """重置所有滤波器"""
        for f in self._filters:
            f.reset()
        if self._validator:
            self._validator.reset()


def create_default_filter() -> CompositeFilter:
    """
    创建默认的组合滤波器

    Returns:
        配置好的组合滤波器
    """
    return (
        CompositeFilter()
        .add_outlier_filter(threshold=3.0, window_size=10)
        .add_kalman_filter(process_variance=1e-5, measure_variance=1e-1)
    )

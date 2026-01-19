"""
滤波模块单元测试
"""

import pytest
import numpy as np

from retrosight.preprocessing.filter import (
    KalmanFilter1D,
    MovingAverage,
    ExponentialSmoothing,
    OutlierFilter,
    ValueValidator,
    CompositeFilter,
    FilterConfig,
    create_default_filter,
)


class TestKalmanFilter1D:
    """卡尔曼滤波器测试"""

    def test_initialization(self):
        """测试初始化"""
        kf = KalmanFilter1D()
        assert kf.process_variance == 1e-5
        assert kf.measure_variance == 1e-1
        assert kf.estimate == 0.0
        assert not kf._initialized

    def test_first_update(self):
        """测试首次更新"""
        kf = KalmanFilter1D()
        result = kf.update(100.0)
        assert result == 100.0
        assert kf._initialized

    def test_smoothing_effect(self):
        """测试平滑效果"""
        kf = KalmanFilter1D()

        # 模拟带噪声的测量值
        measurements = [100.0, 102.0, 98.0, 101.0, 99.0]
        results = [kf.update(m) for m in measurements]

        # 滤波后的值应该比原始值更平滑
        original_variance = np.var(measurements)
        filtered_variance = np.var(results)
        assert filtered_variance < original_variance

    def test_reset(self):
        """测试重置"""
        kf = KalmanFilter1D()
        kf.update(100.0)
        kf.reset(50.0)

        assert kf.estimate == 50.0
        assert not kf._initialized


class TestMovingAverage:
    """滑动平均滤波器测试"""

    def test_initialization(self):
        """测试初始化"""
        ma = MovingAverage(window_size=5)
        assert ma.window_size == 5
        assert len(ma._window) == 0

    def test_single_value(self):
        """测试单个值"""
        ma = MovingAverage(window_size=3)
        result = ma.update(10.0)
        assert result == 10.0

    def test_average_calculation(self):
        """测试平均值计算"""
        ma = MovingAverage(window_size=3)
        ma.update(10.0)
        ma.update(20.0)
        result = ma.update(30.0)
        assert result == 20.0  # (10 + 20 + 30) / 3

    def test_window_sliding(self):
        """测试窗口滑动"""
        ma = MovingAverage(window_size=3)
        ma.update(10.0)
        ma.update(20.0)
        ma.update(30.0)
        result = ma.update(40.0)  # 窗口变为 [20, 30, 40]
        assert result == 30.0  # (20 + 30 + 40) / 3

    def test_value_property(self):
        """测试 value 属性"""
        ma = MovingAverage(window_size=3)
        ma.update(10.0)
        ma.update(20.0)
        assert ma.value == 15.0  # (10 + 20) / 2

    def test_reset(self):
        """测试重置"""
        ma = MovingAverage(window_size=3)
        ma.update(10.0)
        ma.reset()
        assert len(ma._window) == 0


class TestExponentialSmoothing:
    """指数平滑滤波器测试"""

    def test_initialization(self):
        """测试初始化"""
        es = ExponentialSmoothing(alpha=0.3)
        assert es.alpha == 0.3
        assert not es._initialized

    def test_first_update(self):
        """测试首次更新"""
        es = ExponentialSmoothing(alpha=0.3)
        result = es.update(100.0)
        assert result == 100.0
        assert es._initialized

    def test_smoothing_formula(self):
        """测试平滑公式"""
        es = ExponentialSmoothing(alpha=0.5)
        es.update(100.0)
        result = es.update(200.0)
        # new_value = alpha * measurement + (1 - alpha) * old_value
        # = 0.5 * 200 + 0.5 * 100 = 150
        assert result == 150.0

    def test_high_alpha_fast_response(self):
        """测试高 alpha 快速响应"""
        es_fast = ExponentialSmoothing(alpha=0.9)
        es_slow = ExponentialSmoothing(alpha=0.1)

        es_fast.update(0.0)
        es_slow.update(0.0)

        fast_result = es_fast.update(100.0)
        slow_result = es_slow.update(100.0)

        # 高 alpha 应该更接近新值
        assert fast_result > slow_result

    def test_reset(self):
        """测试重置"""
        es = ExponentialSmoothing(alpha=0.3)
        es.update(100.0)
        es.reset(50.0)
        assert es._value == 50.0
        assert not es._initialized


class TestOutlierFilter:
    """异常值过滤器测试"""

    def test_initialization(self):
        """测试初始化"""
        of = OutlierFilter(window_size=10, threshold=3.0)
        assert of.threshold == 3.0
        assert of.min_samples == 3

    def test_initial_samples_pass_through(self):
        """测试初始样本直接通过"""
        of = OutlierFilter(min_samples=3)

        # 前几个样本应该直接返回
        assert of.filter(100.0) == 100.0
        assert of.filter(101.0) == 101.0
        assert of.filter(99.0) == 99.0

    def test_normal_value_passes(self):
        """测试正常值通过"""
        of = OutlierFilter(threshold=3.0, min_samples=3)

        # 建立基准
        for v in [100.0, 101.0, 99.0, 100.5]:
            of.filter(v)

        # 正常值应该通过
        result = of.filter(100.2)
        assert result == 100.2

    def test_outlier_rejected(self):
        """测试异常值被拒绝"""
        of = OutlierFilter(threshold=2.0, min_samples=3)

        # 建立基准（均值约100，标准差很小）
        for v in [100.0, 100.1, 99.9, 100.0]:
            of.filter(v)

        # 异常值应该返回 None
        result = of.filter(200.0)
        assert result is None

    def test_filter_or_last(self):
        """测试 filter_or_last 方法"""
        of = OutlierFilter(threshold=2.0, min_samples=3)

        for v in [100.0, 100.1, 99.9, 100.0]:
            of.filter(v)

        # 异常值应该返回最后一个有效值
        result = of.filter_or_last(200.0)
        assert result == 100.0  # 最后添加到历史的值


class TestValueValidator:
    """数值校验器测试"""

    def test_min_value_validation(self):
        """测试最小值校验"""
        validator = ValueValidator(min_value=0.0)

        valid, value = validator.validate(10.0)
        assert valid is True
        assert value == 10.0

        valid, value = validator.validate(-5.0)
        assert valid is False

    def test_max_value_validation(self):
        """测试最大值校验"""
        validator = ValueValidator(max_value=100.0)

        valid, value = validator.validate(50.0)
        assert valid is True

        valid, value = validator.validate(150.0)
        assert valid is False

    def test_range_validation(self):
        """测试范围校验"""
        validator = ValueValidator(min_value=0.0, max_value=100.0)

        valid, _ = validator.validate(50.0)
        assert valid is True

        valid, _ = validator.validate(-1.0)
        assert valid is False

        valid, _ = validator.validate(101.0)
        assert valid is False

    def test_reset(self):
        """测试重置"""
        validator = ValueValidator(min_value=0.0)
        validator.validate(10.0)
        validator.reset()
        assert validator._last_value is None
        assert validator._last_time is None


class TestCompositeFilter:
    """组合滤波器测试"""

    def test_empty_filter(self):
        """测试空滤波器"""
        cf = CompositeFilter()
        result = cf.filter(100.0)
        assert result == 100.0

    def test_single_filter(self):
        """测试单个滤波器"""
        cf = CompositeFilter()
        cf.add_moving_average(window_size=3)

        cf.filter(10.0)
        cf.filter(20.0)
        result = cf.filter(30.0)
        assert result == 20.0

    def test_chained_filters(self):
        """测试链式滤波器"""
        cf = (
            CompositeFilter()
            .add_outlier_filter(threshold=3.0)
            .add_kalman_filter()
        )

        # 应该能正常处理数据
        result = cf.filter(100.0)
        assert result is not None

    def test_fluent_interface(self):
        """测试流式接口"""
        cf = (
            CompositeFilter()
            .add_kalman_filter()
            .add_moving_average()
            .add_exponential_smoothing()
            .set_validator(min_value=0)
        )

        # 应该返回 CompositeFilter 实例
        assert isinstance(cf, CompositeFilter)

    def test_reset(self):
        """测试重置"""
        cf = CompositeFilter()
        cf.add_moving_average(window_size=3)
        cf.filter(100.0)
        cf.reset()

        # 重置后应该像新的一样
        assert len(cf._filters[0]._window) == 0


class TestCreateDefaultFilter:
    """测试默认滤波器创建"""

    def test_creates_composite_filter(self):
        """测试创建组合滤波器"""
        cf = create_default_filter()
        assert isinstance(cf, CompositeFilter)
        assert len(cf._filters) == 2  # outlier + kalman

    def test_default_filter_works(self):
        """测试默认滤波器正常工作"""
        cf = create_default_filter()

        # 应该能处理正常数据
        result = cf.filter(100.0)
        assert result is not None


class TestFilterConfig:
    """测试滤波器配置"""

    def test_default_values(self):
        """测试默认值"""
        config = FilterConfig()
        assert config.window_size == 5
        assert config.process_variance == 1e-5
        assert config.measure_variance == 1e-1
        assert config.alpha == 0.3
        assert config.outlier_threshold == 3.0

    def test_custom_values(self):
        """测试自定义值"""
        config = FilterConfig(
            window_size=10,
            alpha=0.5,
            min_value=0.0,
            max_value=100.0
        )
        assert config.window_size == 10
        assert config.alpha == 0.5
        assert config.min_value == 0.0
        assert config.max_value == 100.0

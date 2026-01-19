"""
透视变换模块单元测试
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from retrosight.preprocessing.transform import (
    PerspectiveTransform,
    TransformConfig,
    ImageRegistration,
    LensDistortionCorrector,
    four_point_transform,
    auto_perspective_correct,
)


class TestTransformConfig:
    """变换配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = TransformConfig()
        assert config.auto_detect is False
        assert config.target_width == 200
        assert config.target_height == 100

    def test_custom_values(self):
        """测试自定义值"""
        config = TransformConfig(
            auto_detect=True,
            target_width=400,
            target_height=200
        )
        assert config.auto_detect is True
        assert config.target_width == 400


class TestPerspectiveTransform:
    """透视变换测试"""

    def test_initialization(self):
        """测试初始化"""
        transform = PerspectiveTransform()
        assert transform._matrix is None
        assert not transform.is_configured

    def test_set_source_points(self):
        """测试设置源点"""
        transform = PerspectiveTransform()
        points = [(0, 0), (100, 0), (100, 50), (0, 50)]
        transform.set_source_points(points)

        assert transform.is_configured
        assert transform._matrix is not None

    def test_set_source_points_invalid(self):
        """测试无效点数"""
        transform = PerspectiveTransform()

        with pytest.raises(ValueError):
            transform.set_source_points([(0, 0), (100, 0)])  # 只有2个点

    def test_apply_without_config(self):
        """测试未配置时应用"""
        transform = PerspectiveTransform()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = transform.apply(image)
        assert result is not None

    def test_apply_with_config(self):
        """测试配置后应用"""
        config = TransformConfig(target_width=50, target_height=25)
        transform = PerspectiveTransform(config)
        points = [(0, 0), (100, 0), (100, 50), (0, 50)]
        transform.set_source_points(points)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = transform.apply(image)

        assert result.shape == (25, 50, 3)

    def test_transform_point(self):
        """测试点变换"""
        transform = PerspectiveTransform()
        points = [(0, 0), (200, 0), (200, 100), (0, 100)]
        transform.set_source_points(points)

        # 中心点应该映射到目标中心附近
        center = transform.transform_point((100, 50))
        assert center is not None

    def test_order_points(self):
        """测试点排序"""
        transform = PerspectiveTransform()
        # 乱序的点
        points = np.array([[100, 0], [0, 50], [100, 50], [0, 0]])
        ordered = transform._order_points(points)

        # 应该是 [左上, 右上, 右下, 左下]
        assert len(ordered) == 4

    def test_reset(self):
        """测试重置"""
        transform = PerspectiveTransform()
        transform.set_source_points([(0, 0), (100, 0), (100, 50), (0, 50)])
        transform.reset()

        assert not transform.is_configured


class TestImageRegistration:
    """图像配准测试"""

    def test_initialization(self):
        """测试初始化"""
        reg = ImageRegistration()
        assert reg._reference is None

    def test_initialization_methods(self):
        """测试不同初始化方法"""
        for method in ["orb", "akaze"]:
            reg = ImageRegistration(method=method)
            assert reg._detector is not None

    def test_set_reference(self):
        """测试设置参考图像"""
        reg = ImageRegistration()
        ref_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        reg.set_reference(ref_image)

        assert reg._reference is not None

    def test_align_without_reference(self):
        """测试无参考时对齐"""
        reg = ImageRegistration()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = reg.align(image)
        assert result is not None

    def test_reset(self):
        """测试重置"""
        reg = ImageRegistration()
        ref_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        reg.set_reference(ref_image)
        reg.reset()

        assert reg._reference is None


class TestLensDistortionCorrector:
    """镜头畸变校正测试"""

    def test_initialization(self):
        """测试初始化"""
        corrector = LensDistortionCorrector()
        assert not corrector.is_calibrated

    def test_set_coefficients(self):
        """测试设置系数"""
        corrector = LensDistortionCorrector()

        camera_matrix = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

        corrector.set_coefficients(camera_matrix, dist_coeffs)
        assert corrector.is_calibrated

    def test_undistort_without_calibration(self):
        """测试未标定时校正"""
        corrector = LensDistortionCorrector()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = corrector.undistort(image)
        assert np.array_equal(result, image)


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_four_point_transform(self):
        """测试四点变换"""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        points = [(10, 10), (190, 10), (190, 90), (10, 90)]

        result = four_point_transform(image, points, target_size=(100, 50))
        assert result.shape == (50, 100, 3)

    def test_auto_perspective_correct(self):
        """测试自动透视校正"""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = auto_perspective_correct(image)
        assert result is not None

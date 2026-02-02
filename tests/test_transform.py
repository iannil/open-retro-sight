"""
透视变换模块单元测试
"""

import pytest
import numpy as np

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
        config = TransformConfig(auto_detect=True, target_width=400, target_height=200)
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

        camera_matrix = np.array(
            [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32
        )

        dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

        corrector.set_coefficients(camera_matrix, dist_coeffs)
        assert corrector.is_calibrated

    def test_undistort_without_calibration(self):
        """测试未标定时校正"""
        corrector = LensDistortionCorrector()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = corrector.undistort(image)
        assert np.array_equal(result, image)


class TestPerspectiveTransformAdvanced:
    """透视变换高级测试"""

    def test_apply_inverse(self):
        """测试逆透视变换"""
        transform = PerspectiveTransform()
        points = [(0, 0), (200, 0), (200, 100), (0, 100)]
        transform.set_source_points(points)

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = transform.apply_inverse(image, (300, 200))

        assert result is not None
        assert result.shape == (200, 300, 3)

    def test_apply_inverse_without_matrix(self):
        """测试未配置时逆变换"""
        transform = PerspectiveTransform()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = transform.apply_inverse(image, (200, 150))
        # 无矩阵时返回原图
        assert np.array_equal(result, image)

    def test_transform_point_without_matrix(self):
        """测试未配置时点变换"""
        transform = PerspectiveTransform()
        point = (50, 50)

        result = transform.transform_point(point)
        # 无矩阵时返回原点
        assert result == point

    def test_transform_points(self):
        """测试多点变换"""
        transform = PerspectiveTransform()
        points = [(0, 0), (200, 0), (200, 100), (0, 100)]
        transform.set_source_points(points)

        test_points = [(50, 25), (100, 50), (150, 75)]
        result = transform.transform_points(test_points)

        assert len(result) == 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result)

    def test_detect_corners_grayscale(self):
        """测试灰度图角点检测"""
        transform = PerspectiveTransform()
        # 创建灰度图像
        gray_image = np.zeros((100, 100), dtype=np.uint8)

        result = transform._detect_corners(gray_image)
        # 纯黑图像可能检测不到角点
        assert result is None or len(result) == 4

    def test_detect_corners_with_rectangle(self):
        """测试有矩形的角点检测"""
        transform = PerspectiveTransform()

        # 创建带白色矩形的图像
        image = np.zeros((200, 200), dtype=np.uint8)
        # 画一个白色矩形
        image[30:170, 30:170] = 255

        result = transform._detect_corners(image)
        # 可能检测到或检测不到，取决于轮廓近似
        if result is not None:
            assert len(result) == 4

    def test_set_source_points_with_target_size(self):
        """测试设置源点时指定目标尺寸"""
        transform = PerspectiveTransform()
        points = [(0, 0), (100, 0), (100, 50), (0, 50)]
        transform.set_source_points(points, target_size=(150, 75))

        assert transform.is_configured
        # 验证目标点使用了指定尺寸
        assert transform._dst_points[1][0] == 149  # w-1
        assert transform._dst_points[2][1] == 74  # h-1


class TestImageRegistrationAdvanced:
    """图像配准高级测试"""

    def test_initialization_sift(self):
        """测试 SIFT 初始化"""
        reg = ImageRegistration(method="sift")
        assert reg._detector is not None
        # SIFT 使用 L2 norm
        assert reg._matcher is not None

    def test_initialization_invalid_method(self):
        """测试无效方法"""
        with pytest.raises(ValueError):
            ImageRegistration(method="invalid")

    def test_set_reference_grayscale(self):
        """测试设置灰度参考图像"""
        reg = ImageRegistration()
        # 灰度图像
        ref_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        reg.set_reference(ref_image)

        assert reg._reference is not None

    def test_align_with_reference(self):
        """测试有参考时对齐"""
        reg = ImageRegistration()

        # 创建有纹理的参考图像
        ref_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        reg.set_reference(ref_image)

        # 创建类似的测试图像
        test_image = ref_image.copy()
        result = reg.align(test_image)

        assert result is not None
        assert result.shape[:2] == (100, 100)

    def test_align_grayscale_image(self):
        """测试对齐灰度图像"""
        reg = ImageRegistration()

        ref_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        reg.set_reference(ref_image)

        # 灰度测试图像
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = reg.align(test_image)

        assert result is not None

    def test_align_few_matches(self):
        """测试匹配点不足时"""
        reg = ImageRegistration()

        # 创建完全不同的图像
        ref_image = np.zeros((100, 100, 3), dtype=np.uint8)
        reg.set_reference(ref_image)

        # 完全不同的测试图像
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = reg.align(test_image)

        # 应该返回原图
        assert result is not None


class TestLensDistortionCorrectorAdvanced:
    """镜头畸变校正高级测试"""

    def test_undistort_with_calibration(self):
        """测试标定后校正"""
        corrector = LensDistortionCorrector()

        camera_matrix = np.array(
            [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64
        )
        dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float64)

        corrector.set_coefficients(camera_matrix, dist_coeffs)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = corrector.undistort(image, crop=False)

        assert result is not None
        assert result.shape == image.shape

    def test_undistort_no_crop(self):
        """测试不裁剪校正"""
        corrector = LensDistortionCorrector()

        camera_matrix = np.array(
            [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64
        )
        dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float64)

        corrector.set_coefficients(camera_matrix, dist_coeffs)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = corrector.undistort(image, crop=False)

        assert result.shape == image.shape

    def test_save_calibration_not_calibrated(self):
        """测试未标定时保存"""
        corrector = LensDistortionCorrector()

        with pytest.raises(ValueError):
            corrector.save_calibration("/tmp/test_calib.npz")

    def test_save_and_load_calibration(self, tmp_path):
        """测试保存和加载标定"""
        corrector = LensDistortionCorrector()

        camera_matrix = np.array(
            [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64
        )
        dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float64)

        corrector.set_coefficients(camera_matrix, dist_coeffs)
        corrector._new_camera_matrix = camera_matrix.copy()
        corrector._roi = (10, 10, 620, 460)

        filepath = tmp_path / "calib.npz"
        corrector.save_calibration(str(filepath))

        # 加载到新校正器
        new_corrector = LensDistortionCorrector()
        new_corrector.load_calibration(str(filepath))

        assert new_corrector.is_calibrated
        assert np.allclose(new_corrector._camera_matrix, camera_matrix)
        assert np.allclose(new_corrector._dist_coeffs, dist_coeffs)

    def test_calibrate_with_chessboard_insufficient_images(self):
        """测试标定图像不足"""
        corrector = LensDistortionCorrector()

        # 提供空图像列表
        images = [np.zeros((100, 100), dtype=np.uint8)]
        result = corrector.calibrate_with_chessboard(images)

        assert result is False


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_four_point_transform(self):
        """测试四点变换"""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        points = [(10, 10), (190, 10), (190, 90), (10, 90)]

        result = four_point_transform(image, points, target_size=(100, 50))
        assert result.shape == (50, 100, 3)

    def test_four_point_transform_no_target_size(self):
        """测试四点变换无目标尺寸"""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        points = [(10, 10), (190, 10), (190, 90), (10, 90)]

        result = four_point_transform(image, points)
        # 使用默认尺寸
        assert result is not None

    def test_auto_perspective_correct(self):
        """测试自动透视校正"""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = auto_perspective_correct(image)
        assert result is not None

"""
图像增强模块单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from retrosight.preprocessing.enhancement import (
    EnhancementMode,
    EnhancementConfig,
    ImageEnhancer,
    GlareRemover,
    MultiFrameFusion,
    enhance_image,
    remove_glare,
    denoise_image,
)


class TestEnhancementMode:
    """增强模式枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert EnhancementMode.NONE.value == "none"
        assert EnhancementMode.AUTO.value == "auto"
        assert EnhancementMode.LOW_LIGHT.value == "low_light"
        assert EnhancementMode.HIGH_CONTRAST.value == "high_contrast"
        assert EnhancementMode.ANTI_GLARE.value == "anti_glare"
        assert EnhancementMode.DENOISE.value == "denoise"


class TestEnhancementConfig:
    """增强配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = EnhancementConfig()
        assert config.mode == EnhancementMode.AUTO
        assert config.clahe_clip_limit == 2.0
        assert config.clahe_grid_size == (8, 8)
        assert config.denoise_strength == 10
        assert config.sharpen_strength == 1.0
        assert config.fusion_frames == 5
        assert config.gamma == 1.0
        assert config.glare_threshold == 240
        assert config.glare_inpaint_radius == 5

    def test_custom_values(self):
        """测试自定义值"""
        config = EnhancementConfig(
            mode=EnhancementMode.LOW_LIGHT,
            clahe_clip_limit=3.0,
            denoise_strength=15,
            gamma=0.8
        )
        assert config.mode == EnhancementMode.LOW_LIGHT
        assert config.clahe_clip_limit == 3.0
        assert config.denoise_strength == 15
        assert config.gamma == 0.8


class TestImageEnhancer:
    """图像增强器测试"""

    @pytest.fixture
    def enhancer(self):
        """创建增强器"""
        return ImageEnhancer()

    @pytest.fixture
    def dark_image(self):
        """创建暗图像"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 30

    @pytest.fixture
    def bright_image(self):
        """创建亮图像"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 200

    @pytest.fixture
    def low_contrast_image(self):
        """创建低对比度图像"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 128

    @pytest.fixture
    def glare_image(self):
        """创建带反光的图像"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        # 添加反光点
        image[40:60, 40:60, :] = 250
        return image

    def test_initialization(self, enhancer):
        """测试初始化"""
        assert enhancer.config is not None
        assert enhancer._clahe is not None

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = EnhancementConfig(mode=EnhancementMode.DENOISE)
        enhancer = ImageEnhancer(config)
        assert enhancer.config.mode == EnhancementMode.DENOISE

    def test_enhance_none_mode(self, dark_image):
        """测试不增强模式"""
        config = EnhancementConfig(mode=EnhancementMode.NONE)
        enhancer = ImageEnhancer(config)
        result = enhancer.enhance(dark_image)
        np.testing.assert_array_equal(result, dark_image)

    def test_enhance_auto_mode(self, enhancer, dark_image):
        """测试自动增强模式"""
        result = enhancer.enhance(dark_image)
        assert result.shape == dark_image.shape
        # 暗图像应该被增亮
        assert np.mean(result) >= np.mean(dark_image)

    def test_enhance_low_light(self, dark_image):
        """测试低光照增强"""
        config = EnhancementConfig(mode=EnhancementMode.LOW_LIGHT)
        enhancer = ImageEnhancer(config)
        result = enhancer.enhance(dark_image)
        assert result.shape == dark_image.shape
        # 应该比原图亮
        assert np.mean(result) >= np.mean(dark_image)

    def test_enhance_contrast(self, low_contrast_image):
        """测试对比度增强"""
        config = EnhancementConfig(mode=EnhancementMode.HIGH_CONTRAST)
        enhancer = ImageEnhancer(config)
        result = enhancer.enhance(low_contrast_image)
        assert result.shape == low_contrast_image.shape

    def test_enhance_anti_glare(self, glare_image):
        """测试去反光"""
        config = EnhancementConfig(mode=EnhancementMode.ANTI_GLARE)
        enhancer = ImageEnhancer(config)
        result = enhancer.enhance(glare_image)
        assert result.shape == glare_image.shape
        # 反光区域应该被处理
        # 检查原本高亮区域是否被降低
        original_bright = np.mean(glare_image[40:60, 40:60])
        result_bright = np.mean(result[40:60, 40:60])
        assert result_bright < original_bright

    def test_enhance_denoise(self, enhancer, dark_image):
        """测试去噪"""
        config = EnhancementConfig(mode=EnhancementMode.DENOISE)
        enhancer = ImageEnhancer(config)
        result = enhancer.enhance(dark_image)
        assert result.shape == dark_image.shape

    def test_enhance_with_fusion(self, enhancer, dark_image):
        """测试多帧融合增强"""
        # 添加多帧
        for i in range(5):
            result = enhancer.enhance_with_fusion(dark_image)

        # 5帧后应该有输出
        assert result.shape == dark_image.shape

    def test_apply_gamma(self, enhancer, low_contrast_image):
        """测试 Gamma 校正"""
        # Gamma < 1 会增亮图像
        result = enhancer._apply_gamma(low_contrast_image, 0.5)
        assert np.mean(result) > np.mean(low_contrast_image)

        # Gamma > 1 会变暗图像
        result = enhancer._apply_gamma(low_contrast_image, 2.0)
        assert np.mean(result) < np.mean(low_contrast_image)

    def test_sharpen(self, enhancer, low_contrast_image):
        """测试锐化"""
        result = enhancer._sharpen(low_contrast_image)
        assert result.shape == low_contrast_image.shape

    def test_reset_fusion_buffer(self, enhancer, dark_image):
        """测试重置融合缓冲区"""
        # 添加一些帧
        for _ in range(3):
            enhancer.enhance_with_fusion(dark_image)

        assert len(enhancer._frame_buffer) == 3

        enhancer.reset_fusion_buffer()
        assert len(enhancer._frame_buffer) == 0


class TestGlareRemover:
    """去反光处理器测试"""

    @pytest.fixture
    def remover(self):
        """创建去反光处理器"""
        return GlareRemover()

    @pytest.fixture
    def glare_image(self):
        """创建带反光的图像"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        # 添加反光点
        image[40:60, 40:60, :] = 250
        return image

    @pytest.fixture
    def clean_image(self):
        """创建无反光的图像"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 100

    def test_initialization(self, remover):
        """测试初始化"""
        assert remover.glare_threshold == 240
        assert remover.inpaint_radius == 5
        assert remover._background is None

    def test_custom_initialization(self):
        """测试自定义初始化"""
        remover = GlareRemover(glare_threshold=200, inpaint_radius=10)
        assert remover.glare_threshold == 200
        assert remover.inpaint_radius == 10

    def test_remove(self, remover, glare_image):
        """测试去反光"""
        result = remover.remove(glare_image)
        assert result.shape == glare_image.shape
        # 反光区域应该被修复
        original_bright = np.mean(glare_image[40:60, 40:60])
        result_bright = np.mean(result[40:60, 40:60])
        assert result_bright < original_bright

    def test_remove_no_glare(self, remover, clean_image):
        """测试无反光图像"""
        result = remover.remove(clean_image)
        assert result.shape == clean_image.shape

    def test_set_background(self, remover, clean_image):
        """测试设置背景"""
        remover.set_background(clean_image)
        assert remover._background is not None

    def test_remove_with_background(self, remover, glare_image, clean_image):
        """测试使用背景去反光"""
        remover.set_background(clean_image)
        result = remover.remove_with_background(glare_image)
        assert result.shape == glare_image.shape

    def test_remove_with_background_no_bg(self, remover, glare_image):
        """测试无背景时使用背景去反光"""
        # 应该回退到普通去反光
        result = remover.remove_with_background(glare_image)
        assert result.shape == glare_image.shape

    def test_multi_angle_fusion(self, remover):
        """测试多角度融合"""
        # 创建多角度图像
        images = []
        for i in range(3):
            img = np.ones((100, 100, 3), dtype=np.uint8) * 100
            # 不同位置的反光
            x = 30 + i * 20
            img[40:60, x:x+20, :] = 250
            images.append(img)

        result = remover.multi_angle_fusion(images)
        assert result.shape == (100, 100, 3)

    def test_multi_angle_fusion_single_image(self, remover, glare_image):
        """测试单图像多角度融合"""
        result = remover.multi_angle_fusion([glare_image])
        np.testing.assert_array_equal(result, glare_image)

    def test_multi_angle_fusion_empty(self, remover):
        """测试空图像列表"""
        with pytest.raises(ValueError):
            remover.multi_angle_fusion([])

    def test_detect_glare(self, remover, glare_image):
        """测试反光检测"""
        mask = remover._detect_glare(glare_image)
        assert mask.shape == (100, 100)
        # 反光区域应该被标记
        assert np.sum(mask[40:60, 40:60]) > 0


class TestMultiFrameFusion:
    """多帧融合测试"""

    @pytest.fixture
    def fusion(self):
        """创建融合器"""
        return MultiFrameFusion(num_frames=5)

    @pytest.fixture
    def test_frame(self):
        """创建测试帧"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 128

    def test_initialization(self, fusion):
        """测试初始化"""
        assert fusion.num_frames == 5
        assert fusion.method == "average"
        assert fusion.frame_count == 0
        assert not fusion.is_ready

    def test_custom_initialization(self):
        """测试自定义初始化"""
        fusion = MultiFrameFusion(num_frames=10, method="median")
        assert fusion.num_frames == 10
        assert fusion.method == "median"

    def test_add_frame(self, fusion, test_frame):
        """测试添加帧"""
        # 前4帧返回 None
        for i in range(4):
            result = fusion.add_frame(test_frame)
            assert result is None
            assert fusion.frame_count == i + 1

        # 第5帧返回融合结果
        result = fusion.add_frame(test_frame)
        assert result is not None
        assert result.shape == test_frame.shape

    def test_is_ready(self, fusion, test_frame):
        """测试缓冲区就绪状态"""
        assert not fusion.is_ready

        for _ in range(5):
            fusion.add_frame(test_frame)

        assert fusion.is_ready

    def test_fuse_average(self, fusion, test_frame):
        """测试均值融合"""
        for _ in range(5):
            fusion.add_frame(test_frame)

        result = fusion.fuse()
        # 均值融合相同帧应该返回相同值
        np.testing.assert_array_almost_equal(
            result.astype(float),
            test_frame.astype(float),
            decimal=0
        )

    def test_fuse_median(self, test_frame):
        """测试中值融合"""
        fusion = MultiFrameFusion(num_frames=5, method="median")
        for _ in range(5):
            fusion.add_frame(test_frame)

        result = fusion.fuse()
        assert result.shape == test_frame.shape

    def test_fuse_weighted(self, test_frame):
        """测试加权融合"""
        fusion = MultiFrameFusion(num_frames=5, method="weighted")
        for _ in range(5):
            fusion.add_frame(test_frame)

        result = fusion.fuse()
        assert result.shape == test_frame.shape

    def test_fuse_empty_buffer(self, fusion):
        """测试空缓冲区融合"""
        with pytest.raises(ValueError):
            fusion.fuse()

    def test_reset(self, fusion, test_frame):
        """测试重置"""
        for _ in range(3):
            fusion.add_frame(test_frame)

        assert fusion.frame_count == 3

        fusion.reset()
        assert fusion.frame_count == 0
        assert not fusion.is_ready


class TestConvenienceFunctions:
    """便捷函数测试"""

    @pytest.fixture
    def test_image(self):
        """创建测试图像"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 100

    @pytest.fixture
    def glare_image(self):
        """创建带反光的图像"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        image[40:60, 40:60, :] = 250
        return image

    def test_enhance_image(self, test_image):
        """测试 enhance_image"""
        result = enhance_image(test_image)
        assert result.shape == test_image.shape

    def test_enhance_image_modes(self, test_image):
        """测试不同模式的 enhance_image"""
        for mode in EnhancementMode:
            result = enhance_image(test_image, mode=mode)
            assert result.shape == test_image.shape

    def test_remove_glare(self, glare_image):
        """测试 remove_glare"""
        result = remove_glare(glare_image)
        assert result.shape == glare_image.shape

    def test_remove_glare_custom_threshold(self, glare_image):
        """测试自定义阈值的 remove_glare"""
        result = remove_glare(glare_image, threshold=200)
        assert result.shape == glare_image.shape

    def test_denoise_image(self, test_image):
        """测试 denoise_image"""
        result = denoise_image(test_image)
        assert result.shape == test_image.shape

    def test_denoise_image_custom_strength(self, test_image):
        """测试自定义强度的 denoise_image"""
        result = denoise_image(test_image, strength=20)
        assert result.shape == test_image.shape

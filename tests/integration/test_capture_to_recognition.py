"""
采集到识别的集成测试

测试场景:
- 图像采集 → OCR 数字识别
- 图像采集 → 指针识别
- 图像采集 → 指示灯识别
- 图像采集 → 开关识别
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock


class TestCaptureToOCR:
    """采集到 OCR 识别集成测试"""

    def test_image_to_ocr_pipeline(self, sample_digital_image):
        """测试图像到 OCR 的完整流程"""
        from retrosight.recognition.ocr import SimpleOCR

        # 使用简单 OCR（不依赖 PaddleOCR）
        ocr = SimpleOCR()
        result = ocr.recognize(sample_digital_image)

        assert result is not None
        assert hasattr(result, 'text')
        assert hasattr(result, 'confidence')

    def test_preprocessed_image_to_ocr(self, sample_digital_image):
        """测试预处理后的图像识别"""
        from retrosight.recognition.ocr import SimpleOCR
        from retrosight.preprocessing.enhancement import ImageEnhancer, EnhancementConfig

        # 图像增强
        enhancer = ImageEnhancer(EnhancementConfig())
        enhanced = enhancer.enhance(sample_digital_image)

        # OCR 识别
        ocr = SimpleOCR()
        result = ocr.recognize(enhanced)

        assert result is not None

    def test_roi_extraction_to_ocr(self, sample_digital_image):
        """测试 ROI 提取后的识别"""
        from retrosight.recognition.ocr import SimpleOCR

        # 提取 ROI
        roi = sample_digital_image[10:90, 10:190]

        # OCR 识别
        ocr = SimpleOCR()
        result = ocr.recognize(roi)

        assert result is not None


class TestCaptureToPointer:
    """采集到指针识别集成测试"""

    def test_image_to_pointer_pipeline(self, sample_gauge_image):
        """测试图像到指针识别的完整流程"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

        config = GaugeConfig(
            min_value=0,
            max_value=100,
            min_angle=225,
            max_angle=-45,
            unit="MPa"
        )

        recognizer = PointerRecognizer(config)
        result = recognizer.recognize(sample_gauge_image)

        assert result is not None
        assert hasattr(result, 'angle')
        assert hasattr(result, 'value')
        assert hasattr(result, 'confidence')

    def test_calibrated_pointer_recognition(self, sample_gauge_image, temp_calibration_file):
        """测试校准后的指针识别"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig

        recognizer = PointerRecognizer(GaugeConfig())

        # 两点校准
        recognizer.calibrate_two_point(
            angle1=45.0, value1=0.0,
            angle2=315.0, value2=100.0
        )

        # 保存校准
        recognizer.save_calibration(temp_calibration_file)

        # 创建新识别器并加载校准
        new_recognizer = PointerRecognizer(GaugeConfig())
        new_recognizer.load_calibration(temp_calibration_file)

        # 识别
        result = new_recognizer.recognize(sample_gauge_image)

        assert result is not None
        assert new_recognizer.calibration is not None

    def test_transform_then_pointer(self, sample_gauge_image):
        """测试透视变换后的指针识别"""
        from retrosight.recognition.pointer import PointerRecognizer, GaugeConfig
        from retrosight.preprocessing.transform import PerspectiveTransform

        # 应用轻微变换（模拟校正）
        h, w = sample_gauge_image.shape[:2]
        transform = PerspectiveTransform()

        # 设置四点（轻微倾斜）
        src_points = [
            (10, 10),
            (w - 10, 15),
            (w - 15, h - 10),
            (5, h - 15)
        ]
        transform.set_source_points(src_points, w, h)
        corrected = transform.apply(sample_gauge_image)

        # 指针识别
        recognizer = PointerRecognizer(GaugeConfig())
        result = recognizer.recognize(corrected)

        assert result is not None


class TestCaptureToLight:
    """采集到指示灯识别集成测试"""

    def test_image_to_light_detection(self, sample_light_image_green):
        """测试图像到指示灯检测"""
        from retrosight.recognition.light import LightRecognizer, LightConfig, LightColor

        config = LightConfig(
            region=(20, 20, 60, 60),
            expected_colors=[LightColor.GREEN, LightColor.RED]
        )

        recognizer = LightRecognizer(config)
        result = recognizer.detect(sample_light_image_green)

        assert result is not None
        assert hasattr(result, 'color')
        assert hasattr(result, 'state')

    def test_andon_light_detection(self, sample_light_image_green, sample_light_image_red):
        """测试 Andon 灯检测"""
        from retrosight.recognition.light import detect_andon

        # 测试绿灯
        result_green = detect_andon(sample_light_image_green)
        assert result_green is not None

        # 测试红灯
        result_red = detect_andon(sample_light_image_red)
        assert result_red is not None

    def test_enhanced_light_detection(self, sample_light_image_green):
        """测试增强后的指示灯检测"""
        from retrosight.recognition.light import LightRecognizer, LightConfig
        from retrosight.preprocessing.enhancement import enhance_image

        # 图像增强
        enhanced = enhance_image(sample_light_image_green)

        # 检测
        recognizer = LightRecognizer(LightConfig())
        result = recognizer.detect(enhanced)

        assert result is not None


class TestCaptureToSwitch:
    """采集到开关识别集成测试"""

    def test_image_to_switch_detection(self, sample_switch_on_image):
        """测试图像到开关检测"""
        from retrosight.recognition.switch import SwitchRecognizer, SwitchConfig, SwitchType

        config = SwitchConfig(switch_type=SwitchType.TOGGLE)
        recognizer = SwitchRecognizer(config)

        result = recognizer.recognize(sample_switch_on_image)

        assert result is not None
        assert hasattr(result, 'state')

    def test_switch_state_change_detection(self, sample_switch_on_image, sample_switch_off_image):
        """测试开关状态变化检测"""
        from retrosight.recognition.switch import SwitchRecognizer, SwitchConfig, SwitchType

        config = SwitchConfig(switch_type=SwitchType.TOGGLE)
        recognizer = SwitchRecognizer(config)

        result_on = recognizer.recognize(sample_switch_on_image)
        result_off = recognizer.recognize(sample_switch_off_image)

        assert result_on is not None
        assert result_off is not None


class TestPreprocessingPipeline:
    """预处理流水线测试"""

    def test_full_preprocessing_pipeline(self, sample_digital_image):
        """测试完整预处理流水线"""
        from retrosight.preprocessing.enhancement import ImageEnhancer, EnhancementConfig
        from retrosight.preprocessing.filter import create_default_filter

        # 图像增强
        enhancer = ImageEnhancer(EnhancementConfig())
        enhanced = enhancer.enhance(sample_digital_image)

        assert enhanced is not None
        assert enhanced.shape == sample_digital_image.shape

        # 滤波器（用于数值平滑）
        filter_obj = create_default_filter()
        assert filter_obj is not None

        # 模拟连续读数
        values = [10.0, 10.5, 10.2, 10.8, 10.3]
        filtered_values = [filter_obj.filter(v) for v in values]

        assert len(filtered_values) == len(values)

    def test_glare_removal_pipeline(self, sample_digital_image):
        """测试去反光流水线"""
        from retrosight.preprocessing.enhancement import GlareRemover

        # 添加模拟反光
        img_with_glare = sample_digital_image.copy()
        cv2.circle(img_with_glare, (100, 50), 20, (255, 255, 255), -1)

        # 去反光
        remover = GlareRemover(glare_threshold=240)
        result = remover.remove(img_with_glare)

        assert result is not None
        assert result.shape == img_with_glare.shape

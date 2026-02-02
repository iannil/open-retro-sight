"""
OCR 模块单元测试
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock

from retrosight.recognition.ocr import (
    OCRRecognizer,
    OCRConfig,
    OCRResult,
    SimpleOCR,
    recognize_digits,
)


class TestOCRConfig:
    """OCR 配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = OCRConfig()
        assert config.use_gpu is False
        assert config.lang == "en"
        assert config.use_angle_cls is False
        assert config.show_log is False

    def test_custom_values(self):
        """测试自定义值"""
        config = OCRConfig(
            use_gpu=True,
            lang="ch",
            use_angle_cls=True,
            det_model_dir="/path/to/det",
            rec_model_dir="/path/to/rec",
        )
        assert config.use_gpu is True
        assert config.lang == "ch"
        assert config.det_model_dir == "/path/to/det"


class TestOCRResult:
    """OCR 结果测试"""

    def test_creation(self):
        """测试创建"""
        result = OCRResult(text="123.45", value=123.45, confidence=0.95)
        assert result.text == "123.45"
        assert result.value == 123.45
        assert result.confidence == 0.95

    def test_default_values(self):
        """测试默认值"""
        result = OCRResult(text="test")
        assert result.value is None
        assert result.confidence == 0.0
        assert result.bbox is None
        assert result.raw_results == []

    def test_with_bbox(self):
        """测试带边界框"""
        result = OCRResult(text="100", bbox=(10, 20, 50, 30))
        assert result.bbox == (10, 20, 50, 30)


class TestOCRRecognizer:
    """OCR 识别器测试"""

    def test_initialization(self):
        """测试初始化"""
        recognizer = OCRRecognizer()
        assert recognizer._initialized is False
        assert recognizer._ocr is None

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = OCRConfig(lang="ch", use_gpu=True)
        recognizer = OCRRecognizer(config)
        assert recognizer.config.lang == "ch"
        assert recognizer.config.use_gpu is True

    def test_extract_number_integer(self):
        """测试提取整数"""
        recognizer = OCRRecognizer()
        result = recognizer._extract_number("123")
        assert result == 123.0

    def test_extract_number_decimal(self):
        """测试提取小数"""
        recognizer = OCRRecognizer()
        result = recognizer._extract_number("123.45")
        assert result == 123.45

    def test_extract_number_negative(self):
        """测试提取负数"""
        recognizer = OCRRecognizer()
        result = recognizer._extract_number("-50.5")
        assert result == -50.5

    def test_extract_number_scientific(self):
        """测试提取科学计数法"""
        recognizer = OCRRecognizer()
        result = recognizer._extract_number("1.5e3")
        assert result == 1500.0

    def test_extract_number_with_ocr_errors(self):
        """测试 OCR 常见错误修正"""
        recognizer = OCRRecognizer()

        # O -> 0
        assert recognizer._extract_number("1O0") == 100.0

        # l -> 1
        assert recognizer._extract_number("l23") == 123.0

        # , -> .
        assert recognizer._extract_number("12,5") == 12.5

    def test_extract_number_with_spaces(self):
        """测试带空格的数字"""
        recognizer = OCRRecognizer()
        result = recognizer._extract_number("  123  ")
        assert result == 123.0

    def test_extract_number_no_match(self):
        """测试无匹配"""
        recognizer = OCRRecognizer()
        result = recognizer._extract_number("abc")
        assert result is None

    def test_merge_boxes_empty(self):
        """测试空边界框合并"""
        recognizer = OCRRecognizer()
        result = recognizer._merge_boxes([])
        assert result == (0, 0, 0, 0)

    def test_merge_boxes_single(self):
        """测试单个边界框"""
        recognizer = OCRRecognizer()
        boxes = [[[0, 0], [100, 0], [100, 50], [0, 50]]]
        result = recognizer._merge_boxes(boxes)
        assert result == (0, 0, 100, 50)

    def test_merge_boxes_multiple(self):
        """测试多个边界框合并"""
        recognizer = OCRRecognizer()
        boxes = [
            [[0, 0], [50, 0], [50, 30], [0, 30]],
            [[60, 0], [100, 0], [100, 30], [60, 30]],
        ]
        result = recognizer._merge_boxes(boxes)
        x, y, w, h = result
        assert x == 0
        assert y == 0
        assert x + w == 100
        assert y + h == 30

    def test_preprocess_grayscale(self):
        """测试灰度图预处理"""
        recognizer = OCRRecognizer()
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        result = recognizer._preprocess(gray_image)

        # 应该返回 BGR 图像
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_preprocess_color(self):
        """测试彩色图预处理"""
        recognizer = OCRRecognizer()
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognizer._preprocess(color_image)

        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_parse_results_empty(self):
        """测试解析空结果"""
        recognizer = OCRRecognizer()
        result = recognizer._parse_results(None)
        assert result.text == ""
        assert result.confidence == 0.0

    def test_parse_results_empty_list(self):
        """测试解析空列表"""
        recognizer = OCRRecognizer()
        result = recognizer._parse_results([[]])
        assert result.text == ""


class TestSimpleOCR:
    """简化 OCR 测试"""

    def test_initialization_without_tesseract(self):
        """测试无 tesseract 时初始化"""
        with patch.dict("sys.modules", {"pytesseract": None}):
            # 重新导入以触发 ImportError
            SimpleOCR()
            # 即使 tesseract 不可用也不应该抛出异常

    def test_recognize_without_tesseract(self):
        """测试无 tesseract 时识别"""
        ocr = SimpleOCR()
        ocr._tesseract_available = False

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ocr.recognize(image)

        assert result.text == ""
        assert result.confidence == 0.0


class TestRecognizeDigits:
    """便捷函数测试"""

    @patch.object(OCRRecognizer, "recognize")
    def test_recognize_digits_default(self, mock_recognize):
        """测试默认识别"""
        mock_recognize.return_value = OCRResult(text="123", value=123.0)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognize_digits(image, use_simple=False)

        assert result.text == "123"
        mock_recognize.assert_called_once()

    def test_recognize_digits_simple(self):
        """测试简化识别"""
        with patch.object(SimpleOCR, "recognize") as mock_recognize:
            mock_recognize.return_value = OCRResult(text="456", value=456.0)

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            recognize_digits(image, use_simple=True)

            mock_recognize.assert_called_once()


class TestOCRRecognizerInit:
    """OCR 识别器初始化测试"""

    def test_init_ocr_import_error(self):
        """测试 PaddleOCR 未安装时的错误"""
        recognizer = OCRRecognizer()

        with patch.dict("sys.modules", {"paddleocr": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                recognizer._init_ocr()

    def test_init_ocr_already_initialized(self):
        """测试重复初始化"""
        recognizer = OCRRecognizer()
        recognizer._initialized = True

        # 不应该有任何操作
        recognizer._init_ocr()
        assert recognizer._initialized is True


class TestOCRRecognizerParseResults:
    """OCR 结果解析测试"""

    def test_parse_results_with_data(self):
        """测试解析有数据的结果"""
        recognizer = OCRRecognizer()

        # 模拟 PaddleOCR 结果格式
        results = [
            [
                [[[0, 0], [100, 0], [100, 30], [0, 30]], ("123.45", 0.95)],
                [[[0, 35], [50, 35], [50, 60], [0, 60]], ("MPa", 0.90)],
            ]
        ]

        result = recognizer._parse_results(results)

        assert "123.45" in result.text
        assert result.confidence > 0
        assert result.value is not None

    def test_parse_results_single_line(self):
        """测试解析单行结果"""
        recognizer = OCRRecognizer()

        results = [[[[[0, 0], [50, 0], [50, 20], [0, 20]], ("100", 0.99)]]]

        result = recognizer._parse_results(results)

        assert result.text == "100"
        assert abs(result.confidence - 0.99) < 0.01


class TestOCRRecognizerExtractNumber:
    """数值提取测试"""

    def test_extract_number_value_error(self):
        """测试无效数字格式"""
        recognizer = OCRRecognizer()

        # 这种情况不太可能发生，但测试边界条件
        result = recognizer._extract_number("abc-def")
        assert result is None

    def test_extract_number_with_ocr_errors_combined(self):
        """测试多种 OCR 错误组合"""
        recognizer = OCRRecognizer()

        # O 和 l 组合
        result = recognizer._extract_number("lO0")
        assert result == 100.0

        # I 和空格
        result = recognizer._extract_number("I 23")
        assert result == 123.0


class TestSimpleOCRAdvanced:
    """SimpleOCR 高级测试"""

    def test_initialization_import_error(self):
        """测试 pytesseract 导入错误"""
        with patch.dict("sys.modules", {"pytesseract": None}):
            with patch("importlib.util.find_spec", return_value=None):
                ocr = SimpleOCR()
                assert ocr._tesseract_available is False

    def test_recognize_with_tesseract_available(self):
        """测试 tesseract 可用时的识别"""
        ocr = SimpleOCR()

        # 模拟 tesseract 可用
        ocr._tesseract_available = True

        # 模拟 pytesseract
        mock_pytesseract = Mock()
        mock_pytesseract.image_to_string = Mock(return_value="123.45")

        with patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = ocr.recognize(image)

            assert result.text == "123.45"
            assert result.value == 123.45
            assert result.confidence == 0.8

    def test_recognize_with_tesseract_empty_result(self):
        """测试 tesseract 空结果"""
        ocr = SimpleOCR()
        ocr._tesseract_available = True

        mock_pytesseract = Mock()
        mock_pytesseract.image_to_string = Mock(return_value="")

        with patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = ocr.recognize(image)

            assert result.text == ""
            assert result.confidence == 0.0

    def test_recognize_with_tesseract_invalid_number(self):
        """测试 tesseract 返回无效数字"""
        ocr = SimpleOCR()
        ocr._tesseract_available = True

        mock_pytesseract = Mock()
        mock_pytesseract.image_to_string = Mock(return_value="abc")

        with patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = ocr.recognize(image)

            assert result.text == "abc"
            assert result.value is None

    def test_recognize_grayscale_image(self):
        """测试灰度图像识别"""
        ocr = SimpleOCR()
        ocr._tesseract_available = True

        mock_pytesseract = Mock()
        mock_pytesseract.image_to_string = Mock(return_value="100")

        with patch.dict("sys.modules", {"pytesseract": mock_pytesseract}):
            # 灰度图像
            image = np.zeros((100, 100), dtype=np.uint8)
            result = ocr.recognize(image)

            assert result.text == "100"


class TestOCRRecognizerRegion:
    """区域识别测试"""

    @patch.object(OCRRecognizer, "recognize")
    def test_recognize_region(self, mock_recognize):
        """测试区域识别"""
        mock_recognize.return_value = OCRResult(
            text="100", value=100.0, bbox=(5, 5, 20, 10)
        )

        recognizer = OCRRecognizer()
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        # 识别区域 (50, 50, 100, 50)
        result = recognizer.recognize_region(image, (50, 50, 100, 50))

        # 边界框应该被转换为原图坐标
        assert result.bbox == (55, 55, 20, 10)

    @patch.object(OCRRecognizer, "recognize")
    def test_recognize_region_no_bbox(self, mock_recognize):
        """测试区域识别无边界框"""
        mock_recognize.return_value = OCRResult(text="100", value=100.0, bbox=None)

        recognizer = OCRRecognizer()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        region = (50, 50, 100, 50)

        result = recognizer.recognize_region(image, region)

        # 应该使用区域作为边界框
        assert result.bbox == region

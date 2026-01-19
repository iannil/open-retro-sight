"""
数字OCR识别模块

支持识别类型：
- 七段数码管 (LED/LCD)
- 普通数字屏幕
- 混合文字数字

基于 PaddleOCR 实现，针对工业场景优化
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str                           # 识别文本
    value: Optional[float] = None       # 解析后的数值
    confidence: float = 0.0             # 置信度 (0-1)
    bbox: Optional[Tuple[int, int, int, int]] = None  # 边界框 (x, y, w, h)
    raw_results: list = field(default_factory=list)   # 原始OCR结果


@dataclass
class OCRConfig:
    """OCR配置"""
    use_gpu: bool = False               # 是否使用GPU
    lang: str = "en"                    # 语言 (en/ch)
    det_model_dir: Optional[str] = None # 检测模型路径
    rec_model_dir: Optional[str] = None # 识别模型路径
    use_angle_cls: bool = False         # 是否使用方向分类
    show_log: bool = False              # 是否显示日志


class OCRRecognizer:
    """
    数字OCR识别器

    使用示例:
    ```python
    recognizer = OCRRecognizer()

    # 识别图像中的数字
    result = recognizer.recognize(image)
    print(f"识别结果: {result.text}, 数值: {result.value}")

    # 识别指定区域
    result = recognizer.recognize_region(image, (100, 100, 200, 50))
    ```
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        """
        初始化OCR识别器

        Args:
            config: OCR配置
        """
        self.config = config or OCRConfig()
        self._ocr = None
        self._initialized = False

    def _init_ocr(self):
        """延迟初始化OCR引擎"""
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_angle_cls=self.config.use_angle_cls,
                lang=self.config.lang,
                use_gpu=self.config.use_gpu,
                show_log=self.config.show_log,
                det_model_dir=self.config.det_model_dir,
                rec_model_dir=self.config.rec_model_dir,
            )
            self._initialized = True
            logger.info("PaddleOCR 初始化成功")

        except ImportError:
            logger.error("PaddleOCR 未安装，请运行: pip install paddleocr paddlepaddle")
            raise

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        识别图像中的数字

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            OCR识别结果
        """
        self._init_ocr()

        # 预处理图像
        processed = self._preprocess(image)

        # OCR识别
        results = self._ocr.ocr(processed, cls=self.config.use_angle_cls)

        # 解析结果
        return self._parse_results(results)

    def recognize_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> OCRResult:
        """
        识别指定区域的数字

        Args:
            image: 输入图像
            region: 区域 (x, y, width, height)

        Returns:
            OCR识别结果
        """
        x, y, w, h = region

        # 裁剪区域
        cropped = image[y:y+h, x:x+w]

        # 识别
        result = self.recognize(cropped)

        # 更新边界框为原图坐标
        if result.bbox:
            bx, by, bw, bh = result.bbox
            result.bbox = (x + bx, y + by, bw, bh)
        else:
            result.bbox = region

        return result

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理

        针对七段数码管优化：
        - 灰度化
        - 对比度增强
        - 二值化
        - 形态学处理（增强小数点）
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 对比度增强 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 自适应二值化
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # 形态学处理 - 增强小数点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        processed = cv2.dilate(binary, kernel, iterations=1)

        # 转回BGR（PaddleOCR需要）
        result = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        return result

    def _parse_results(self, results: list) -> OCRResult:
        """
        解析OCR结果

        Args:
            results: PaddleOCR原始结果

        Returns:
            解析后的结果
        """
        if not results or not results[0]:
            return OCRResult(text="", confidence=0.0)

        # 合并所有识别文本
        texts = []
        confidences = []
        boxes = []

        for line in results[0]:
            if len(line) >= 2:
                box, (text, conf) = line[0], line[1]
                texts.append(text)
                confidences.append(conf)
                boxes.append(box)

        # 合并文本
        full_text = " ".join(texts)

        # 计算平均置信度
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # 提取数值
        value = self._extract_number(full_text)

        # 计算边界框
        bbox = self._merge_boxes(boxes) if boxes else None

        return OCRResult(
            text=full_text,
            value=value,
            confidence=avg_conf,
            bbox=bbox,
            raw_results=results
        )

    def _extract_number(self, text: str) -> Optional[float]:
        """
        从文本中提取数值

        支持格式：
        - 整数: 123
        - 小数: 123.45
        - 负数: -123.45
        - 科学计数法: 1.23e5
        """
        # 清理文本
        cleaned = text.strip()

        # 替换常见OCR错误
        cleaned = cleaned.replace('O', '0')
        cleaned = cleaned.replace('o', '0')
        cleaned = cleaned.replace('l', '1')
        cleaned = cleaned.replace('I', '1')
        cleaned = cleaned.replace(',', '.')
        cleaned = cleaned.replace(' ', '')

        # 正则匹配数字
        pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, cleaned)

        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass

        return None

    def _merge_boxes(
        self,
        boxes: List[List[List[float]]]
    ) -> Tuple[int, int, int, int]:
        """
        合并多个边界框

        Args:
            boxes: 边界框列表 (每个框是4个角点)

        Returns:
            合并后的边界框 (x, y, w, h)
        """
        if not boxes:
            return (0, 0, 0, 0)

        all_points = []
        for box in boxes:
            for point in box:
                all_points.append(point)

        all_points = np.array(all_points)

        x_min = int(np.min(all_points[:, 0]))
        y_min = int(np.min(all_points[:, 1]))
        x_max = int(np.max(all_points[:, 0]))
        y_max = int(np.max(all_points[:, 1]))

        return (x_min, y_min, x_max - x_min, y_max - y_min)


class SimpleOCR:
    """
    简化版OCR（不依赖PaddleOCR，用于测试）

    使用 Tesseract 或简单模板匹配
    """

    def __init__(self):
        self._tesseract_available = False
        try:
            import pytesseract
            self._tesseract_available = True
        except ImportError:
            logger.warning("pytesseract 未安装，SimpleOCR 功能受限")

    def recognize(self, image: np.ndarray) -> OCRResult:
        """识别图像中的数字"""
        if not self._tesseract_available:
            return OCRResult(text="", confidence=0.0)

        import pytesseract

        # 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Tesseract识别
        config = '--psm 7 -c tessedit_char_whitelist=0123456789.-'
        text = pytesseract.image_to_string(binary, config=config).strip()

        # 提取数值
        value = None
        try:
            value = float(text) if text else None
        except ValueError:
            pass

        return OCRResult(text=text, value=value, confidence=0.8 if text else 0.0)


def recognize_digits(image: np.ndarray, use_simple: bool = False) -> OCRResult:
    """
    便捷函数：识别图像中的数字

    Args:
        image: 输入图像
        use_simple: 是否使用简化版OCR

    Returns:
        识别结果
    """
    if use_simple:
        return SimpleOCR().recognize(image)
    return OCRRecognizer().recognize(image)

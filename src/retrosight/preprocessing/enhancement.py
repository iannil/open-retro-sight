"""
图像增强模块

功能：
- 去反光处理：消除玻璃/屏幕反光干扰
- 多帧融合：提升信噪比和稳定性
- 自适应对比度增强：改善低光照条件
- 去噪：减少图像噪点
- 锐化：增强边缘清晰度

用于改善恶劣环境下的识别效果
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Deque
from dataclasses import dataclass
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class EnhancementMode(Enum):
    """增强模式"""
    NONE = "none"                 # 不增强
    AUTO = "auto"                 # 自动选择
    LOW_LIGHT = "low_light"       # 低光照增强
    HIGH_CONTRAST = "high_contrast"  # 高对比度
    ANTI_GLARE = "anti_glare"     # 去反光
    DENOISE = "denoise"           # 去噪


@dataclass
class EnhancementConfig:
    """增强配置"""
    # 增强模式
    mode: EnhancementMode = EnhancementMode.AUTO
    # CLAHE 对比度限制
    clahe_clip_limit: float = 2.0
    # CLAHE 网格大小
    clahe_grid_size: Tuple[int, int] = (8, 8)
    # 去噪强度 (0-30)
    denoise_strength: int = 10
    # 锐化强度 (0-3)
    sharpen_strength: float = 1.0
    # 多帧融合帧数
    fusion_frames: int = 5
    # Gamma 校正值
    gamma: float = 1.0
    # 反光检测阈值 (亮度)
    glare_threshold: int = 240
    # 反光修复半径
    glare_inpaint_radius: int = 5


class ImageEnhancer:
    """
    图像增强器

    使用示例:
    ```python
    # 基本使用
    enhancer = ImageEnhancer()
    enhanced = enhancer.enhance(image)

    # 去反光模式
    config = EnhancementConfig(mode=EnhancementMode.ANTI_GLARE)
    enhancer = ImageEnhancer(config)
    enhanced = enhancer.enhance(image)

    # 多帧融合
    enhancer = ImageEnhancer()
    for frame in video_frames:
        enhanced = enhancer.enhance_with_fusion(frame)
    ```
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        """
        初始化图像增强器

        Args:
            config: 增强配置
        """
        self.config = config or EnhancementConfig()
        self._frame_buffer: Deque[np.ndarray] = deque(
            maxlen=self.config.fusion_frames
        )
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        增强图像

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            增强后的图像
        """
        if self.config.mode == EnhancementMode.NONE:
            return image

        if self.config.mode == EnhancementMode.AUTO:
            return self._auto_enhance(image)
        elif self.config.mode == EnhancementMode.LOW_LIGHT:
            return self._enhance_low_light(image)
        elif self.config.mode == EnhancementMode.HIGH_CONTRAST:
            return self._enhance_contrast(image)
        elif self.config.mode == EnhancementMode.ANTI_GLARE:
            return self._remove_glare(image)
        elif self.config.mode == EnhancementMode.DENOISE:
            return self._denoise(image)
        else:
            return image

    def enhance_with_fusion(self, image: np.ndarray) -> np.ndarray:
        """
        使用多帧融合增强图像

        Args:
            image: 当前帧

        Returns:
            融合增强后的图像
        """
        # 添加到帧缓冲
        self._frame_buffer.append(image.copy())

        # 执行帧融合
        fused = self._temporal_fusion()

        # 再执行空间增强
        return self.enhance(fused)

    def _auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """自动选择增强方法"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 分析图像特征
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # 检测是否有反光
        has_glare = np.sum(gray > self.config.glare_threshold) > (
            gray.size * 0.01
        )

        result = image.copy()

        # 低光照
        if mean_brightness < 50:
            result = self._enhance_low_light(result)

        # 低对比度
        if std_brightness < 30:
            result = self._enhance_contrast(result)

        # 有反光
        if has_glare:
            result = self._remove_glare(result)

        # Gamma 校正
        if self.config.gamma != 1.0:
            result = self._apply_gamma(result, self.config.gamma)

        return result

    def _enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """低光照增强"""
        # 转换到 LAB 色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 对亮度通道应用 CLAHE
        l_enhanced = self._clahe.apply(l)

        # 合并通道
        lab_enhanced = cv2.merge([l_enhanced, a, b])

        # 转回 BGR
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 可选: 应用 Gamma 校正进一步提亮
        if self.config.gamma < 1.0:
            result = self._apply_gamma(result, 0.7)

        return result

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """对比度增强"""
        # 使用 CLAHE 增强对比度
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 增强亮度通道
        l_enhanced = self._clahe.apply(l)

        # 合并
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 可选: 锐化
        if self.config.sharpen_strength > 0:
            result = self._sharpen(result)

        return result

    def _remove_glare(self, image: np.ndarray) -> np.ndarray:
        """
        去除反光

        使用多种技术组合：
        1. 检测高亮区域
        2. 图像修复（inpainting）填充反光区域
        3. 自适应阈值处理
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测反光区域（高亮点）
        _, glare_mask = cv2.threshold(
            gray,
            self.config.glare_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # 膨胀反光区域，确保完全覆盖
        kernel = np.ones((5, 5), np.uint8)
        glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)

        # 使用图像修复填充反光区域
        result = cv2.inpaint(
            image,
            glare_mask,
            self.config.glare_inpaint_radius,
            cv2.INPAINT_TELEA
        )

        return result

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """去噪"""
        # 使用非局部均值去噪
        if len(image.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                self.config.denoise_strength,
                self.config.denoise_strength,
                7,
                21
            )
        else:
            result = cv2.fastNlMeansDenoising(
                image,
                None,
                self.config.denoise_strength,
                7,
                21
            )

        return result

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """锐化"""
        # 使用 Unsharp Masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(
            image,
            1 + self.config.sharpen_strength,
            gaussian,
            -self.config.sharpen_strength,
            0
        )
        return sharpened

    def _apply_gamma(
        self,
        image: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """应用 Gamma 校正"""
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")

        return cv2.LUT(image, table)

    def _temporal_fusion(self) -> np.ndarray:
        """
        时域融合（多帧平均）

        减少噪声，提高稳定性
        """
        if len(self._frame_buffer) == 0:
            raise ValueError("Frame buffer is empty")

        if len(self._frame_buffer) == 1:
            return self._frame_buffer[0]

        # 转换为浮点数进行平均
        frames = [f.astype(np.float32) for f in self._frame_buffer]

        # 加权平均（最新帧权重更高）
        weights = np.array([
            0.5 ** (len(frames) - 1 - i)
            for i in range(len(frames))
        ])
        weights = weights / np.sum(weights)

        result = np.zeros_like(frames[0])
        for i, (frame, weight) in enumerate(zip(frames, weights)):
            result += frame * weight

        return result.astype(np.uint8)

    def reset_fusion_buffer(self):
        """重置融合缓冲区"""
        self._frame_buffer.clear()


class GlareRemover:
    """
    专用去反光处理器

    针对玻璃/屏幕反光的高级处理

    使用示例:
    ```python
    remover = GlareRemover()

    # 方法1: 单帧去反光
    result = remover.remove(image)

    # 方法2: 使用参考背景
    remover.set_background(background_image)
    result = remover.remove_with_background(image)

    # 方法3: 多角度融合
    images = [capture_from_angle(a) for a in angles]
    result = remover.multi_angle_fusion(images)
    ```
    """

    def __init__(
        self,
        glare_threshold: int = 240,
        inpaint_radius: int = 5
    ):
        """
        初始化去反光处理器

        Args:
            glare_threshold: 反光检测亮度阈值
            inpaint_radius: 修复半径
        """
        self.glare_threshold = glare_threshold
        self.inpaint_radius = inpaint_radius
        self._background: Optional[np.ndarray] = None

    def remove(self, image: np.ndarray) -> np.ndarray:
        """
        去除反光（单帧）

        Args:
            image: 输入图像

        Returns:
            去反光后的图像
        """
        # 检测反光
        glare_mask = self._detect_glare(image)

        if cv2.countNonZero(glare_mask) == 0:
            return image

        # 使用图像修复
        result = cv2.inpaint(
            image,
            glare_mask,
            self.inpaint_radius,
            cv2.INPAINT_TELEA
        )

        return result

    def set_background(self, background: np.ndarray):
        """
        设置参考背景

        Args:
            background: 无反光的背景图像
        """
        self._background = background.copy()

    def remove_with_background(self, image: np.ndarray) -> np.ndarray:
        """
        使用参考背景去反光

        通过与背景比较来识别和替换反光区域

        Args:
            image: 输入图像

        Returns:
            去反光后的图像
        """
        if self._background is None:
            return self.remove(image)

        # 确保尺寸匹配
        if image.shape != self._background.shape:
            self._background = cv2.resize(
                self._background,
                (image.shape[1], image.shape[0])
            )

        # 计算差异
        diff = cv2.absdiff(image, self._background)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 检测反光（与背景差异大且亮度高的区域）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(
            gray, self.glare_threshold, 255, cv2.THRESH_BINARY
        )
        _, diff_mask = cv2.threshold(
            diff_gray, 30, 255, cv2.THRESH_BINARY
        )

        # 反光区域 = 高亮度 & 与背景差异大
        glare_mask = cv2.bitwise_and(bright_mask, diff_mask)

        # 膨胀
        kernel = np.ones((5, 5), np.uint8)
        glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)

        # 用背景替换反光区域
        result = image.copy()
        result[glare_mask > 0] = self._background[glare_mask > 0]

        return result

    def multi_angle_fusion(
        self,
        images: List[np.ndarray]
    ) -> np.ndarray:
        """
        多角度图像融合

        使用不同角度拍摄的图像，选择每个区域反光最少的像素

        Args:
            images: 多角度图像列表

        Returns:
            融合后的图像
        """
        if len(images) == 0:
            raise ValueError("No images provided")

        if len(images) == 1:
            return images[0]

        # 确保所有图像尺寸相同
        base_shape = images[0].shape
        aligned_images = []
        for img in images:
            if img.shape != base_shape:
                img = cv2.resize(img, (base_shape[1], base_shape[0]))
            aligned_images.append(img)

        # 计算每张图像的亮度
        brightness_maps = []
        for img in aligned_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness_maps.append(gray)

        # 选择每个像素亮度最低的图像（避免反光）
        brightness_stack = np.stack(brightness_maps, axis=0)
        min_indices = np.argmin(brightness_stack, axis=0)

        # 构建结果图像
        result = np.zeros_like(aligned_images[0])
        for i in range(len(aligned_images)):
            mask = (min_indices == i)
            for c in range(3):
                result[:, :, c][mask] = aligned_images[i][:, :, c][mask]

        return result

    def _detect_glare(self, image: np.ndarray) -> np.ndarray:
        """检测反光区域"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高亮区域
        _, mask = cv2.threshold(
            gray,
            self.glare_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # 膨胀确保完全覆盖
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask


class MultiFrameFusion:
    """
    多帧融合处理器

    通过融合多帧提高图像质量

    使用示例:
    ```python
    fusion = MultiFrameFusion(num_frames=10)

    for frame in video_stream:
        # 添加帧并获取融合结果
        result = fusion.add_frame(frame)
        if result is not None:
            process(result)
    ```
    """

    def __init__(
        self,
        num_frames: int = 5,
        method: str = "average"
    ):
        """
        初始化多帧融合器

        Args:
            num_frames: 融合帧数
            method: 融合方法 ("average", "median", "weighted")
        """
        self.num_frames = num_frames
        self.method = method
        self._buffer: Deque[np.ndarray] = deque(maxlen=num_frames)

    def add_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        添加帧并返回融合结果

        Args:
            frame: 输入帧

        Returns:
            融合结果（缓冲区满时）或 None
        """
        self._buffer.append(frame.copy())

        if len(self._buffer) < self.num_frames:
            return None

        return self.fuse()

    def fuse(self) -> np.ndarray:
        """
        执行帧融合

        Returns:
            融合后的图像
        """
        if len(self._buffer) == 0:
            raise ValueError("No frames in buffer")

        if self.method == "average":
            return self._average_fusion()
        elif self.method == "median":
            return self._median_fusion()
        elif self.method == "weighted":
            return self._weighted_fusion()
        else:
            return self._average_fusion()

    def _average_fusion(self) -> np.ndarray:
        """均值融合"""
        frames = [f.astype(np.float32) for f in self._buffer]
        result = np.mean(frames, axis=0)
        return result.astype(np.uint8)

    def _median_fusion(self) -> np.ndarray:
        """中值融合（对异常值更鲁棒）"""
        frames = np.stack(list(self._buffer), axis=0)
        result = np.median(frames, axis=0)
        return result.astype(np.uint8)

    def _weighted_fusion(self) -> np.ndarray:
        """加权融合（最新帧权重更高）"""
        frames = [f.astype(np.float32) for f in self._buffer]
        n = len(frames)

        # 指数权重
        weights = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        weights = weights / np.sum(weights)

        result = np.zeros_like(frames[0])
        for frame, weight in zip(frames, weights):
            result += frame * weight

        return result.astype(np.uint8)

    def reset(self):
        """重置缓冲区"""
        self._buffer.clear()

    @property
    def is_ready(self) -> bool:
        """缓冲区是否已满"""
        return len(self._buffer) >= self.num_frames

    @property
    def frame_count(self) -> int:
        """当前帧数"""
        return len(self._buffer)


def enhance_image(
    image: np.ndarray,
    mode: EnhancementMode = EnhancementMode.AUTO
) -> np.ndarray:
    """
    便捷函数：增强图像

    Args:
        image: 输入图像
        mode: 增强模式

    Returns:
        增强后的图像
    """
    config = EnhancementConfig(mode=mode)
    enhancer = ImageEnhancer(config)
    return enhancer.enhance(image)


def remove_glare(
    image: np.ndarray,
    threshold: int = 240
) -> np.ndarray:
    """
    便捷函数：去除反光

    Args:
        image: 输入图像
        threshold: 反光检测阈值

    Returns:
        去反光后的图像
    """
    remover = GlareRemover(glare_threshold=threshold)
    return remover.remove(image)


def denoise_image(
    image: np.ndarray,
    strength: int = 10
) -> np.ndarray:
    """
    便捷函数：去噪

    Args:
        image: 输入图像
        strength: 去噪强度

    Returns:
        去噪后的图像
    """
    config = EnhancementConfig(
        mode=EnhancementMode.DENOISE,
        denoise_strength=strength
    )
    enhancer = ImageEnhancer(config)
    return enhancer.enhance(image)

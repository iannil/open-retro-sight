"""
预处理模块

图像预处理功能:
- 透视变换: 校正摄像头角度
- 图像配准: 抗振动对齐
- 数据平滑: 卡尔曼滤波/滑动平均
- 图像增强: 去反光/去噪/对比度增强
"""

from retrosight.preprocessing.filter import (
    FilterConfig,
    KalmanFilter1D,
    MovingAverage,
    ExponentialSmoothing,
    OutlierFilter,
    ValueValidator,
    CompositeFilter,
    create_default_filter,
)

from retrosight.preprocessing.transform import (
    TransformConfig,
    PerspectiveTransform,
    ImageRegistration,
    LensDistortionCorrector,
    four_point_transform,
    auto_perspective_correct,
)

from retrosight.preprocessing.enhancement import (
    EnhancementConfig,
    EnhancementMode,
    ImageEnhancer,
    GlareRemover,
    MultiFrameFusion,
    enhance_image,
    remove_glare,
    denoise_image,
)

__all__ = [
    # filter
    "FilterConfig",
    "KalmanFilter1D",
    "MovingAverage",
    "ExponentialSmoothing",
    "OutlierFilter",
    "ValueValidator",
    "CompositeFilter",
    "create_default_filter",
    # transform
    "TransformConfig",
    "PerspectiveTransform",
    "ImageRegistration",
    "LensDistortionCorrector",
    "four_point_transform",
    "auto_perspective_correct",
    # enhancement
    "EnhancementConfig",
    "EnhancementMode",
    "ImageEnhancer",
    "GlareRemover",
    "MultiFrameFusion",
    "enhance_image",
    "remove_glare",
    "denoise_image",
]

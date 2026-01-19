#!/usr/bin/env python3
"""
Open-RetroSight 示例 01: 基础 OCR 识别

功能说明:
- 创建测试图像（模拟七段数码管显示）
- 使用 SimpleOCR 进行数字识别
- 显示识别结果

依赖:
- opencv-python-headless
- numpy

运行方法:
    python examples/01_basic_ocr.py
"""

import numpy as np
import cv2


def create_test_image(text: str = "123.45") -> np.ndarray:
    """
    创建测试图像

    Args:
        text: 要显示的文本

    Returns:
        模拟的数码管图像
    """
    # 创建深灰色背景
    img = np.zeros((100, 250, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)

    # 绘制绿色数字（模拟数码管）
    cv2.putText(
        img,
        text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),  # 绿色
        3
    )

    return img


def basic_ocr_example():
    """基础 OCR 示例"""
    print("=" * 50)
    print("Open-RetroSight 基础 OCR 示例")
    print("=" * 50)

    # 导入 OCR 模块
    from retrosight.recognition.ocr import SimpleOCR

    # 创建测试图像
    print("\n1. 创建测试图像...")
    test_image = create_test_image("456.78")
    print(f"   图像尺寸: {test_image.shape}")

    # 初始化 OCR
    print("\n2. 初始化 SimpleOCR...")
    ocr = SimpleOCR()
    print("   OCR 引擎就绪")

    # 执行识别
    print("\n3. 执行 OCR 识别...")
    result = ocr.recognize(test_image)

    # 输出结果
    print("\n4. 识别结果:")
    print(f"   文本: {result.text}")
    print(f"   置信度: {result.confidence:.2%}")

    # 尝试解析数值
    print("\n5. 数值解析:")
    try:
        value = float(result.text) if result.text else None
        if value is not None:
            print(f"   解析值: {value}")
        else:
            print("   无法解析为数值")
    except ValueError:
        print(f"   无法将 '{result.text}' 解析为数值")

    print("\n" + "=" * 50)
    print("示例完成")
    print("=" * 50)


def batch_ocr_example():
    """批量 OCR 示例"""
    print("\n" + "=" * 50)
    print("批量 OCR 识别示例")
    print("=" * 50)

    from retrosight.recognition.ocr import SimpleOCR

    ocr = SimpleOCR()

    # 测试不同的数值
    test_values = ["123", "45.6", "789.01", "0.00", "999"]

    print("\n识别结果:")
    print("-" * 40)
    print(f"{'输入':^10} | {'识别':^10} | {'置信度':^10}")
    print("-" * 40)

    for value in test_values:
        image = create_test_image(value)
        result = ocr.recognize(image)
        print(f"{value:^10} | {result.text:^10} | {result.confidence:^10.2%}")

    print("-" * 40)


def ocr_with_preprocessing():
    """带预处理的 OCR 示例"""
    print("\n" + "=" * 50)
    print("带预处理的 OCR 示例")
    print("=" * 50)

    from retrosight.recognition.ocr import SimpleOCR
    from retrosight.preprocessing.enhancement import ImageEnhancer, EnhancementConfig

    # 创建测试图像
    test_image = create_test_image("123.45")

    # 添加噪声模拟真实场景
    noise = np.random.normal(0, 10, test_image.shape).astype(np.uint8)
    noisy_image = cv2.add(test_image, noise)

    print("\n1. 原始图像（带噪声）")

    # 预处理
    print("2. 应用图像增强...")
    enhancer = ImageEnhancer(EnhancementConfig(
        denoise=True,
        sharpen=True
    ))
    enhanced_image = enhancer.enhance(noisy_image)
    print("   增强完成")

    # OCR 识别
    print("3. 执行 OCR 识别...")
    ocr = SimpleOCR()

    result_noisy = ocr.recognize(noisy_image)
    result_enhanced = ocr.recognize(enhanced_image)

    print("\n4. 对比结果:")
    print(f"   噪声图像: {result_noisy.text} (置信度: {result_noisy.confidence:.2%})")
    print(f"   增强图像: {result_enhanced.text} (置信度: {result_enhanced.confidence:.2%})")


if __name__ == "__main__":
    # 运行基础示例
    basic_ocr_example()

    # 运行批量示例
    batch_ocr_example()

    # 运行预处理示例
    ocr_with_preprocessing()

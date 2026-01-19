# 核心算法

## 概述

Open-RetroSight 的核心是一系列图像处理和机器学习算法，用于从摄像头画面中提取设备数据。

## 图像预处理

### 透视变换（Perspective Transformation）

**目的**：允许摄像头侧拍，软件自动"拉正"图像

**原理**：
```
   原始图像（梯形）          校正后（矩形）
   ╱────────────╲           ┌──────────┐
  ╱              ╲    →     │          │
 ╱                ╲         │          │
╱──────────────────╲        └──────────┘
```

**实现**：
```python
import cv2
import numpy as np

# 源点（梯形四角）
src_points = np.float32([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
# 目标点（矩形四角）
dst_points = np.float32([[0,0], [w,0], [w,h], [0,h]])

# 计算变换矩阵
M = cv2.getPerspectiveTransform(src_points, dst_points)
# 应用变换
result = cv2.warpPerspective(img, M, (w, h))
```

### 图像配准（Image Registration）

**目的**：应对设备振动，保持识别区域稳定

**原理**：
1. 检测固定特征点（Logo、边框、螺丝孔等）
2. 每帧图像与参考帧对齐
3. 消除振动导致的偏移

**实现**：
```python
# 特征点检测
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(ref_img, None)
kp2, des2 = orb.detectAndCompute(cur_img, None)

# 特征匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 计算变换矩阵并对齐
```

## 数字识别算法

### PaddleOCR 流程

```
原始图像 → 预处理 → 文本检测 → 文本识别 → 后处理 → 结果
```

**预处理**：
- 灰度化
- 二值化
- 降噪

**针对七段数码管优化**：
- 使用专门的训练数据微调
- PP-OCRv4 模型效果最佳

### 小数点处理

**问题**：小数点经常被忽略

**解决方案**：

1. **形态学处理**
```python
# 膨胀操作，增强小数点
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=1)
```

2. **逻辑校验**
```python
def validate_reading(value, min_val, max_val, last_value, max_change):
    # 范围检查
    if value < min_val or value > max_val:
        return last_value
    # 突变检查
    if abs(value - last_value) > max_change:
        return last_value
    return value
```

## 指针识别算法

### 流程
```
原始图像 → 表盘定位(YOLO) → 指针检测(Hough) → 角度计算 → 数值映射
```

### YOLOv8-Nano 表盘定位

**功能**：检测并裁剪表盘区域

```python
from ultralytics import YOLO

model = YOLO('gauge_detector.pt')
results = model(image)
# 获取表盘边界框
bbox = results[0].boxes[0].xyxy
```

### Hough 变换指针检测

**原理**：在图像中检测直线

```python
# 边缘检测
edges = cv2.Canny(gray, 50, 150)

# Hough 变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                        minLineLength=30, maxLineGap=10)

# 找到最长的线（指针）
pointer_line = max(lines, key=lambda l: line_length(l))
```

### 角度计算与映射

```python
def angle_to_value(angle, zero_angle, full_angle, min_val, max_val):
    """将指针角度映射为数值"""
    angle_range = full_angle - zero_angle
    value_range = max_val - min_val

    relative_angle = angle - zero_angle
    value = min_val + (relative_angle / angle_range) * value_range

    return value
```

## 数据平滑算法

### 卡尔曼滤波

**目的**：平滑识别结果，过滤噪声

```python
class KalmanFilter1D:
    def __init__(self, process_var=1e-5, measure_var=1e-1):
        self.process_var = process_var
        self.measure_var = measure_var
        self.estimate = 0
        self.error = 1

    def update(self, measurement):
        # 预测
        prediction = self.estimate
        pred_error = self.error + self.process_var

        # 更新
        gain = pred_error / (pred_error + self.measure_var)
        self.estimate = prediction + gain * (measurement - prediction)
        self.error = (1 - gain) * pred_error

        return self.estimate
```

### 滑动平均

**目的**：简单有效的平滑方法

```python
from collections import deque

class MovingAverage:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def update(self, value):
        self.window.append(value)
        return sum(self.window) / len(self.window)
```

### 异常值过滤

```python
def filter_outlier(value, history, threshold=3):
    """基于标准差的异常值过滤"""
    if len(history) < 3:
        return value

    mean = np.mean(history)
    std = np.std(history)

    if abs(value - mean) > threshold * std:
        return mean  # 返回历史均值
    return value
```

## 多帧合成

### LED屏刷新率问题

**问题**：某些老LED屏有扫描频率，单帧拍摄可能残缺

**解决方案**：
```python
def multi_frame_fusion(frames, method='max'):
    """多帧融合"""
    if method == 'max':
        return np.max(frames, axis=0)
    elif method == 'mean':
        return np.mean(frames, axis=0).astype(np.uint8)
    elif method == 'median':
        return np.median(frames, axis=0).astype(np.uint8)
```

## 相关文档

- [架构总览](overview.md)
- [软件技术栈](software-stack.md)
- [技术挑战](../roadmap/challenges.md)

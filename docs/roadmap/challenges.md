# 技术挑战与对策

## 概述

Open-RetroSight 在实际工业环境中会面临诸多技术挑战，本文档详细分析各类问题及解决方案。

## 挑战一：光线变化

### 问题描述
- 白天阳光直射导致反光
- 晚上光线不足导致图像过暗
- 光线变化导致识别失效

### 解决方案

#### 1. 软件层面
- **自动曝光算法**：根据画面亮度自动调整曝光参数
- **直方图均衡化**：增强对比度
- **自适应阈值**：动态调整二值化阈值

```python
# 自适应阈值示例
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
```

#### 2. 硬件层面
- **环形补光灯**：提供稳定、均匀的光源
- **遮光罩**：隔绝环境光干扰
- **偏振滤镜**：减少反光（高端方案）

#### 3. 部署建议
- 避免摄像头正对光源
- 选择光线相对稳定的安装位置
- 必要时搭建简易遮光装置

## 挑战二：微小振动

### 问题描述
- 机器运作时产生振动
- 振动导致摄像头画面抖动
- 识别选区偏移，影响准确性

### 解决方案

#### 图像配准算法（Image Registration）

**原理**：每一帧都先根据固定的特征点对齐画面，再进行识别。

**特征点选择**：
- Logo
- 边框角点
- 螺丝孔
- 固定标识

**实现步骤**：
1. 在参考帧中标记特征点
2. 在当前帧中检测对应特征点
3. 计算变换矩阵
4. 对齐当前帧到参考帧

```python
# ORB 特征匹配示例
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(ref_frame, None)
kp2, des2 = orb.detectAndCompute(current_frame, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)

# 计算单应性矩阵并对齐
```

#### 硬件层面
- 使用减震支架
- 摄像头固定要牢固
- 选择防抖摄像头（OIS）

## 挑战三：屏幕刷新率

### 问题描述
- 某些老 LED 屏采用扫描显示
- 摄像头快门速度与扫描频率不同步
- 拍出来的画面是残缺的（只有部分数字亮）

### 解决方案

#### 1. 多帧合成技术

**原理**：采集多帧图像，合成完整画面

```python
def capture_and_merge(cap, num_frames=5):
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    # 取最大值合成（确保所有亮的部分都被捕获）
    merged = np.max(frames, axis=0)
    return merged
```

#### 2. 硬件方案
- 使用支持**高帧率**的摄像头（60fps+）
- 使用**全局快门**摄像头（Global Shutter）
- 调整曝光时间，覆盖完整扫描周期

#### 3. 参数调优
- 降低快门速度（增加曝光时间）
- 使用手动曝光模式

## 挑战四：识别精度

### 问题描述
- 小数点经常被忽略
- "10.5" 被识别成 "105"
- 相似数字混淆（如 8 和 3、6 和 0）

### 解决方案

#### 1. 形态学处理优化

**针对小数点**：
```python
# 膨胀操作增强小数点
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilated = cv2.dilate(binary, kernel, iterations=1)
```

#### 2. 逻辑判断校验

**合理阈值**：
```python
def validate_temperature(value, last_value):
    # 温度不可能瞬间跳到 1000 度
    if value > 500:
        return last_value

    # 温度变化不可能超过 50 度/秒
    if abs(value - last_value) > 50:
        return last_value

    return value
```

**范围检查**：
- 设定数据的合理范围
- 超出范围视为异常

#### 3. 置信度过滤
- OCR 返回置信度
- 低于阈值的结果使用历史值

#### 4. 多次识别投票
```python
def recognize_with_voting(image, times=3):
    results = []
    for _ in range(times):
        result = ocr.recognize(image)
        results.append(result)

    # 取众数
    return max(set(results), key=results.count)
```

## 挑战五：多种设备适配

### 问题描述
- 不同设备的显示屏/仪表差异大
- 通用模型难以覆盖所有情况

### 解决方案

#### 1. 用户自定义校准
- 提供校准流程
- 用户标注样本
- 在线学习优化

#### 2. 模型微调
- 收集特定设备的样本
- 微调 OCR 模型
- 提供专用模型下载

#### 3. 规则引擎
- 支持用户配置识别规则
- 正则表达式匹配
- 自定义后处理逻辑

## 挑战总结

| 挑战 | 软件方案 | 硬件方案 | 复杂度 |
|-----|---------|---------|--------|
| 光线变化 | 自动曝光、直方图均衡 | 补光灯、遮光罩 | 中 |
| 微小振动 | 图像配准算法 | 减震支架 | 中 |
| 屏幕刷新率 | 多帧合成 | 高帧率/全局快门摄像头 | 高 |
| 识别精度 | 形态学处理、逻辑校验 | - | 中 |
| 设备适配 | 模型微调、规则引擎 | - | 高 |

## 相关文档

- [核心算法](../architecture/algorithms.md)
- [硬件层](../architecture/hardware-layer.md)
- [MVP功能](mvp-features.md)

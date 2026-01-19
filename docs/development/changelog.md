# 变更日志

本文档记录 Open-RetroSight 项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [0.1.0] - 2025-01-19

### 🎉 首个功能完整版本

完成 MVP 全部功能开发，包含 Phase 1-3 所有模块。

### 新增

#### 图像采集 (`capture/`)
- `Camera` 类：支持 USB/CSI/RTSP/视频文件
- 多线程帧缓冲，降低延迟
- 可配置分辨率、帧率、曝光

#### 数字识别 (`recognition/ocr.py`)
- `OCRRecognizer`：PaddleOCR 集成
- `SimpleOCR`：轻量级七段数码管识别
- 图像预处理优化

#### 指针识别 (`recognition/pointer.py`)
- `PointerRecognizer`：Hough 线变换检测
- 自动表盘中心检测
- 角度-数值线性映射

#### 指示灯识别 (`recognition/light.py`)
- `LightRecognizer`：HSV 颜色检测
- `AndonMonitor`：Andon 灯 + OEE 计算
- 闪烁检测与频率估算

#### 开关识别 (`recognition/switch.py`)
- `SwitchRecognizer`：拨动/旋钮/按钮/滑动开关
- 模板匹配 + 位置检测
- `MultiSwitchMonitor`：多开关监控

#### 数据滤波 (`preprocessing/filter.py`)
- `KalmanFilter1D`：卡尔曼滤波
- `MovingAverage`：滑动平均
- `ExponentialSmoothing`：指数平滑
- `OutlierFilter`：异常值过滤
- `CompositeFilter`：复合滤波器

#### 透视变换 (`preprocessing/transform.py`)
- `PerspectiveTransform`：四点透视变换
- `ImageRegistration`：ORB/SIFT 图像配准
- `LensDistortionCorrector`：镜头畸变校正

#### 图像增强 (`preprocessing/enhancement.py`)
- `ImageEnhancer`：CLAHE、低光照增强、去噪
- `GlareRemover`：反光检测 + inpainting 修复
- `MultiFrameFusion`：多帧时域融合

#### MQTT 输出 (`output/mqtt.py`)
- `MQTTPublisher`：消息发布
- `MQTTSubscriber`：消息订阅
- 自动重连、离线缓存

#### Modbus TCP (`output/modbus.py`)
- `ModbusServer`：TCP Server（伪装 PLC）
- 多数据类型支持
- 自动寄存器分配

#### 断网续传 (`output/buffer.py`)
- `PersistentBuffer`：SQLite 持久化
- `StoreAndForward`：存储转发
- 优先级队列、自动重试

#### Web 界面 (`ui/app.py`)
- Streamlit 配置界面
- 实时视频预览
- ROI 区域配置

### 测试
- 新增 12 个测试文件
- 覆盖所有核心模块
- 57+ 测试类

### 文档
- 完善项目 README
- 更新开发进度文档
- API 参考文档

---

## [未发布]

### 计划中
- 集成测试
- 实机验证（树莓派）
- Docker 部署
- 性能优化

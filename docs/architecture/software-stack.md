# 软件技术栈

## 概述

Open-RetroSight 采用成熟的开源技术栈，确保稳定性和可维护性。

## 技术栈总览

```
┌─────────────────────────────────────────────────────────────┐
│                      配置界面层                              │
│                 Streamlit / Vue.js                          │
├─────────────────────────────────────────────────────────────┤
│                      通讯协议层                              │
│              MQTT Client / Modbus TCP Server                │
├─────────────────────────────────────────────────────────────┤
│                      核心算法层                              │
│        PaddleOCR / Tesseract / YOLOv8-Nano                  │
├─────────────────────────────────────────────────────────────┤
│                      图像处理层                              │
│                       OpenCV                                │
├─────────────────────────────────────────────────────────────┤
│                      运行环境                                │
│                   Python 3.8+                               │
└─────────────────────────────────────────────────────────────┘
```

## 图像采集：OpenCV

### 功能
- 视频流获取与处理
- 去抖动
- 自动白平衡
- 图像预处理

### 版本要求
- OpenCV 4.5+
- opencv-python-headless（无GUI环境）

### 主要模块
```python
import cv2

# 视频流获取
cap = cv2.VideoCapture(0)

# 图像处理
cv2.cvtColor()      # 颜色空间转换
cv2.threshold()     # 二值化
cv2.Canny()         # 边缘检测
cv2.warpPerspective() # 透视变换
```

## 数字识别

### PaddleOCR（推荐）

**特点**：
- PP-OCRv4 效果极佳
- 针对七段数码管可微调
- 支持中英文
- 轻量级，适合边缘设备

**版本要求**：
- PaddlePaddle 2.4+
- PaddleOCR 2.6+

### Tesseract（备选）

**特点**：
- 纯开源
- 成熟稳定
- 资源占用低

**版本要求**：
- Tesseract 4.0+
- pytesseract

## 指针识别：YOLOv8-Nano

### 功能
- 目标检测定位表盘
- 轻量级，适合边缘设备

### 版本要求
- ultralytics 8.0+

### 配合算法
- **Hough 变换**：检测直线角度
- **几何计算**：角度到数值映射

## 通讯协议

### MQTT

**用途**：发往云端/平台

**特点**：
- 轻量级发布-订阅协议
- 适合物联网场景
- 支持断线重连

**推荐库**：
- paho-mqtt

**数据格式**：
```json
{
  "device_id": "device_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "readings": [
    {"name": "temperature", "value": 180.5, "unit": "°C"},
    {"name": "pressure", "value": 2.5, "unit": "MPa"}
  ]
}
```

### Modbus TCP Server

**用途**：伪装成PLC，供SCADA系统抓取

**特点**：
- 工业标准协议
- 兼容现有SCADA系统
- 无缝集成

**推荐库**：
- pymodbus

**寄存器映射**：
| 寄存器地址 | 数据 | 说明 |
|-----------|------|------|
| 40001 | 温度×10 | 整数表示，需除10 |
| 40002 | 压力×100 | 整数表示，需除100 |
| 40003 | 状态码 | 0=故障, 1=待机, 2=运行 |

## 配置界面

### Streamlit（推荐原型）

**特点**：
- 纯 Python 开发
- 快速原型
- 实时预览

**适用场景**：
- 开发调试
- 小规模部署

### Vue.js（推荐生产）

**特点**：
- 专业前端框架
- 响应式设计
- 高性能

**配合后端**：
- FastAPI / Flask

## 依赖管理

### requirements.txt 示例
```
opencv-python-headless>=4.5.0
paddlepaddle>=2.4.0
paddleocr>=2.6.0
ultralytics>=8.0.0
paho-mqtt>=1.6.0
pymodbus>=3.0.0
streamlit>=1.20.0
numpy>=1.20.0
```

## 部署环境

### Python 版本
- 推荐：Python 3.8 - 3.10
- 最低：Python 3.7

### 操作系统
- Linux（推荐）
- Windows
- macOS（开发环境）

## 相关文档

- [架构总览](overview.md)
- [硬件层](hardware-layer.md)
- [核心算法](algorithms.md)

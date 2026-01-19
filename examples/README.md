# Open-RetroSight 使用示例

本目录包含 Open-RetroSight 的使用示例，帮助您快速上手各项功能。

## 示例列表

| 示例 | 功能 | 依赖 |
|-----|------|------|
| [01_basic_ocr.py](01_basic_ocr.py) | 基础 OCR 识别 | opencv, numpy |
| [02_pointer_gauge.py](02_pointer_gauge.py) | 指针仪表读取 | opencv, numpy |
| [03_indicator_light.py](03_indicator_light.py) | 指示灯监控 | opencv, numpy |
| [04_mqtt_output.py](04_mqtt_output.py) | MQTT 数据输出 | paho-mqtt |
| [05_modbus_server.py](05_modbus_server.py) | Modbus TCP 服务 | pymodbus |
| [06_full_pipeline.py](06_full_pipeline.py) | 完整流程示例 | 全部 |

## 运行前准备

### 安装依赖

```bash
# 安装基础依赖
pip install open-retrosight

# 安装可选依赖（Modbus 支持）
pip install open-retrosight[modbus]

# 安装全部依赖
pip install open-retrosight[all]
```

### 准备测试图像

示例使用程序生成的测试图像，无需额外准备。如需使用实际图像：

```python
import cv2

# 从摄像头采集
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# 从文件读取
image = cv2.imread("your_image.png")
```

## 运行示例

```bash
# 运行基础 OCR 示例
python examples/01_basic_ocr.py

# 运行指针仪表示例
python examples/02_pointer_gauge.py

# 运行完整流程示例
python examples/06_full_pipeline.py
```

## 常见问题

### Q: PaddleOCR 初始化很慢？

A: 首次运行时会下载模型文件，后续运行会使用缓存。如需离线使用，请提前下载模型。

### Q: 摄像头打不开？

A: 检查摄像头设备号，尝试 `cv2.VideoCapture(1)` 或其他设备号。

### Q: MQTT 连接失败？

A: 确保 MQTT Broker 已启动，检查主机名和端口配置。

## 更多资源

- [完整文档](../docs/)
- [API 参考](../docs/api/)
- [配置指南](../docs/configuration.md)

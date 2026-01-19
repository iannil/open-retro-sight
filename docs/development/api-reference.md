# API 参考

> 本文档提供 Open-RetroSight 核心 API 的快速参考。

---

## 图像采集

### Camera

```python
from retrosight.capture import Camera, CameraConfig

# 创建配置
config = CameraConfig(
    source=0,                # 摄像头索引或 RTSP URL
    width=640,
    height=480,
    fps=30
)

# 使用摄像头
with Camera(config) as cam:
    frame = cam.read()
    if frame is not None:
        process(frame)
```

### 便捷函数

```python
from retrosight.capture import capture_single_frame, list_cameras

# 单帧采集
frame = capture_single_frame(source=0)

# 列举摄像头
cameras = list_cameras()
```

---

## 数字识别

### OCRRecognizer

```python
from retrosight.recognition import OCRRecognizer, OCRConfig

config = OCRConfig(
    use_gpu=False,
    preprocess=True
)
recognizer = OCRRecognizer(config)

result = recognizer.recognize(image)
print(f"识别结果: {result.text}, 置信度: {result.confidence}")
```

### SimpleOCR (七段数码管)

```python
from retrosight.recognition import SimpleOCR

ocr = SimpleOCR()
result = ocr.recognize(image)
```

### 便捷函数

```python
from retrosight.recognition import recognize_digits

result = recognize_digits(image, use_simple=False)
```

---

## 指针识别

### PointerRecognizer

```python
from retrosight.recognition import PointerRecognizer, GaugeConfig

config = GaugeConfig(
    min_value=0,
    max_value=10,
    min_angle=-135,
    max_angle=135,
    unit="MPa"
)
recognizer = PointerRecognizer(config)

result = recognizer.recognize(image)
print(f"读数: {result.value} {config.unit}")
```

### 便捷函数

```python
from retrosight.recognition import recognize_gauge

result = recognize_gauge(
    image,
    min_value=0,
    max_value=100,
    unit="°C"
)
```

---

## 指示灯识别

### LightRecognizer

```python
from retrosight.recognition import LightRecognizer, LightConfig

config = LightConfig(
    region=(100, 100, 50, 50),  # ROI
    brightness_threshold=100
)
recognizer = LightRecognizer(config)

result = recognizer.detect(image)
print(f"颜色: {result.color}, 状态: {result.state}")
```

### AndonMonitor (OEE 计算)

```python
from retrosight.recognition import AndonMonitor

monitor = AndonMonitor()

while True:
    frame = camera.read()
    status = monitor.update(frame)
    print(f"可用率: {status['availability']:.2%}")
```

### 便捷函数

```python
from retrosight.recognition import detect_light, detect_andon

# 单灯检测
result = detect_light(image, region=(x, y, w, h))

# Andon 灯检测
andon = detect_andon(image)
# 返回: {"red": "on/off", "yellow": "on/off", "green": "on/off"}
```

---

## 开关识别

### SwitchRecognizer

```python
from retrosight.recognition import SwitchRecognizer, SwitchConfig, SwitchType

# 拨动开关
config = SwitchConfig(switch_type=SwitchType.TOGGLE)
recognizer = SwitchRecognizer(config)
result = recognizer.recognize(image)
print(f"状态: {result.state}")

# 旋钮
config = SwitchConfig(
    switch_type=SwitchType.ROTARY,
    num_positions=4,
    position_labels=["OFF", "LOW", "MED", "HIGH"]
)
recognizer = SwitchRecognizer(config)
result = recognizer.recognize(image)
print(f"档位: {result.position_label}")
```

### 便捷函数

```python
from retrosight.recognition import detect_switch, detect_rotary

result = detect_switch(image, switch_type=SwitchType.TOGGLE)
result = detect_rotary(image, num_positions=3)
```

---

## 数据滤波

### 滤波器

```python
from retrosight.preprocessing import (
    KalmanFilter1D,
    MovingAverage,
    ExponentialSmoothing,
    OutlierFilter,
    CompositeFilter
)

# 卡尔曼滤波
kf = KalmanFilter1D(process_noise=0.01, measurement_noise=0.1)
filtered = kf.filter(raw_value)

# 滑动平均
ma = MovingAverage(window_size=5)
filtered = ma.filter(raw_value)

# 复合滤波器
composite = CompositeFilter()
composite.add_filter(OutlierFilter(max_change=10))
composite.add_filter(MovingAverage(window_size=3))
filtered = composite.filter(raw_value)
```

### 便捷函数

```python
from retrosight.preprocessing import create_default_filter

filter = create_default_filter()  # 异常值过滤 + 卡尔曼 + 滑动平均
```

---

## 透视变换

### PerspectiveTransform

```python
from retrosight.preprocessing import PerspectiveTransform

transform = PerspectiveTransform()

# 设置源点和目标尺寸
transform.set_source_points(
    [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
    output_width=200,
    output_height=200
)

# 应用变换
corrected = transform.apply(image)
```

### ImageRegistration (抗振动)

```python
from retrosight.preprocessing import ImageRegistration

registration = ImageRegistration()
registration.set_reference(ref_image)

aligned = registration.align(current_frame)
```

### 便捷函数

```python
from retrosight.preprocessing import four_point_transform

result = four_point_transform(image, points)
```

---

## 图像增强

### ImageEnhancer

```python
from retrosight.preprocessing import ImageEnhancer, EnhancementConfig, EnhancementMode

config = EnhancementConfig(
    mode=EnhancementMode.AUTO,
    clahe_clip_limit=2.0
)
enhancer = ImageEnhancer(config)

enhanced = enhancer.enhance(image)
```

### GlareRemover

```python
from retrosight.preprocessing import GlareRemover

remover = GlareRemover(glare_threshold=240)
result = remover.remove(image)

# 使用背景参考
remover.set_background(background_image)
result = remover.remove_with_background(image)
```

### 便捷函数

```python
from retrosight.preprocessing import enhance_image, remove_glare, denoise_image

enhanced = enhance_image(image, mode=EnhancementMode.LOW_LIGHT)
deglared = remove_glare(image, threshold=240)
denoised = denoise_image(image, strength=10)
```

---

## MQTT 输出

### MQTTPublisher

```python
from retrosight.output import MQTTPublisher, MQTTConfig, SensorData

config = MQTTConfig(
    broker="localhost",
    port=1883,
    client_id="gateway_001"
)

publisher = MQTTPublisher(config)
publisher.connect()

# 发布数据
data = SensorData(
    sensor_id="temp_01",
    value=25.5,
    unit="°C"
)
publisher.publish(data)

publisher.disconnect()
```

---

## Modbus TCP

### ModbusServer

```python
from retrosight.output import ModbusServer, ModbusConfig, RegisterMapping, DataType

config = ModbusConfig(host="0.0.0.0", port=502)
server = ModbusServer(config)

# 添加寄存器映射
server.add_mapping(RegisterMapping(
    sensor_id="temp_01",
    address=0,
    data_type=DataType.FLOAT32
))

# 启动服务器
server.start()

# 更新数值
server.update_value("temp_01", 25.5)
```

---

## 断网续传

### StoreAndForward

```python
from retrosight.output import StoreAndForward, BufferConfig

def send_func(topic, payload):
    # 实际发送逻辑
    return mqtt_client.publish(topic, payload)

config = BufferConfig(
    storage_path="buffer.db",
    max_size_mb=100
)

saf = StoreAndForward(send_func=send_func, config=config)
saf.send("topic", "payload")

# 网络恢复时处理积压
saf.set_online(True)
saf.flush()
```

---

## Web 界面

### 启动应用

```bash
# 命令行
streamlit run src/retrosight/ui/app.py

# 或通过入口点
retrosight
```

---

## 相关文档

- [开发进度](progress.md)
- [变更日志](changelog.md)
- [架构总览](../architecture/overview.md)

# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

Open-RetroSight 是一款非侵入式工业边缘AI网关软件，通过计算机视觉将传统"哑设备"的数据数字化。核心价值主张："不拆机、不停产、不改线，给老机器装上'数字眼睛'。"

## 项目状态

当前阶段：**MVP v0.1.0 完成** - Phase 1-3 全部实现

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -e ".[dev]"

# 安装所有可选依赖
pip install -e ".[all]"

# 运行测试
pytest

# 代码格式化
black src/ tests/

# 代码检查
ruff check src/ tests/

# 启动 Web 界面
streamlit run src/retrosight/ui/app.py
```

## 代码结构

```text
src/retrosight/
├── __init__.py              # 包入口
├── capture/                 # 图像采集模块
│   └── camera.py           # 摄像头视频流获取
├── recognition/             # 识别模块
│   ├── ocr.py              # 数字OCR识别
│   ├── pointer.py          # 指针仪表识别
│   ├── light.py            # 指示灯识别
│   └── switch.py           # 开关/旋钮识别
├── preprocessing/           # 预处理模块
│   ├── filter.py           # 数据平滑
│   ├── transform.py        # 透视变换
│   └── enhancement.py      # 图像增强/去反光
├── output/                  # 输出模块
│   ├── mqtt.py             # MQTT发布
│   ├── modbus.py           # Modbus TCP服务
│   └── buffer.py           # 断网续传缓存
└── ui/                      # Web界面
    └── app.py              # Streamlit应用
```

## 文档结构

```text
docs/
├── index.md                 # 文档首页/导航
├── getting-started/         # 入门指南
├── use-cases/               # 应用场景
├── architecture/            # 技术架构
├── roadmap/                 # 产品规划
└── development/             # 开发文档
    ├── progress.md          # 开发进度
    ├── api-reference.md     # API参考
    └── changelog.md         # 变更日志
```

## 目标应用场景

1. 七段数码管/LCD屏幕读取 - 读取温度、压力、计数等数据
2. 指针式仪表读取 - 通过指针角度检测读取压力表、流量计
3. 状态指示灯识别 - Andon灯识别，用于OEE计算
4. 物理开关/旋钮位置识别 - 控制面板状态检测

## 技术栈

- 图像采集：OpenCV
- 数字识别：PaddleOCR
- 指针识别：YOLOv8-Nano + Hough变换
- 通讯协议：MQTT、Modbus TCP
- 配置界面：Streamlit

## 目标硬件平台

- Raspberry Pi Zero 2W / 香橙派 + USB工业摄像头
- 退役Android手机
- ESP32-CAM + 边缘服务器

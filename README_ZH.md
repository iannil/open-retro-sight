# Open-Retro-Sight

> 非侵入式工业边缘AI网关 - 给老机器装上"数字眼睛"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[English](README.md)

## 简介

Open-RetroSight 是一款非侵入式的工业边缘AI网关软件，通过计算机视觉技术将传统"哑设备"的数据数字化。

**不拆机、不停产、不改线** —— 用几百元成本，几分钟部署，实现老旧设备的数字化改造。

## 核心能力

- **七段数码管/LCD屏幕识别** - 读取温度、压力、计数等数字
- **指针式仪表读取** - 通过指针角度检测映射为数值
- **状态指示灯识别** - Andon灯识别，计算OEE
- **开关/旋钮位置检测** - 识别档位状态

## 快速开始

```bash
# 克隆项目
git clone https://github.com/open-retrosight/open-retrosight.git
cd open-retrosight

# 安装依赖
pip install -r requirements.txt

# 或使用 pip 安装
pip install -e .
```

## 文档

详细文档请查看 [docs/](docs/index.md)：

- [项目介绍](docs/getting-started/introduction.md) - 背景、价值、适用场景
- [技术架构](docs/architecture/overview.md) - 系统设计与技术栈
- [应用场景](docs/use-cases/) - 各类设备的识别方案
- [产品规划](docs/roadmap/) - 愿景、MVP功能、商业化

## 硬件要求

- Raspberry Pi Zero 2W / 香橙派 + USB摄像头
- 或 退役Android手机
- 或 ESP32-CAM + 边缘服务器

## 技术栈

`Python` `OpenCV` `PaddleOCR` `YOLOv8` `MQTT` `Modbus TCP` `Streamlit`

## 项目状态

当前阶段：**MVP v0.1.0 完成**

### 已完成里程碑

- [x] 产品规划与设计
- [x] 文档体系建立
- [x] 项目结构初始化
- [x] Phase 1: 基础能力（视频流、OCR、MQTT、Web界面）
- [x] Phase 2: 核心功能（指针识别、透视校正、Modbus、断网续传）
- [x] Phase 3: 扩展功能（指示灯、开关识别、图像增强）

### 发展路线

| 阶段 | 重点 | 状态 |
|------|------|------|
| **v0.1** | MVP 核心功能 | ✅ 完成 |
| **v0.2** | 社区建设 | 计划中 |
| **v0.3** | 硬件套件与产品打磨 | 计划中 |
| **v0.4** | 云平台上线（SaaS） | 计划中 |
| **v1.0** | 生态完善与算法市场 | 计划中 |

#### 近期计划 (v0.2)
- [ ] 建立开发者社区
- [ ] 完善文档和教程
- [ ] 收集用户反馈、持续迭代
- [ ] 增加更多设备识别模板

#### 未来规划
- **硬件套件**：开箱即用的摄像头、补光灯、防护外壳套装
- **高级算法**：复杂仪表识别、单摄像头多目标检测
- **云平台**：多设备管理、数据存储、报警推送
- **行业解决方案**：制造业、能源、化工、物流等垂直领域

### Phase 1 已实现功能

- **视频流采集** (`src/retrosight/capture/camera.py`)
  - USB/CSI/RTSP 摄像头支持
  - 多线程帧缓冲

- **数字 OCR 识别** (`src/retrosight/recognition/ocr.py`)
  - 七段数码管识别 (PaddleOCR)
  - 图像预处理优化

- **数据平滑滤波** (`src/retrosight/preprocessing/filter.py`)
  - 卡尔曼滤波、滑动平均、指数平滑
  - 异常值过滤

- **MQTT 数据发布** (`src/retrosight/output/mqtt.py`)
  - 断线重连、离线缓存
  - 结构化 JSON 数据

- **Web 配置界面** (`src/retrosight/ui/app.py`)
  - Streamlit 实时预览
  - ROI 区域配置

### Phase 2 已实现功能

- **透视变换校正** (`src/retrosight/preprocessing/transform.py`)
  - 四点透视变换
  - 图像配准（抗振动）
  - 镜头畸变校正

- **指针识别算法** (`src/retrosight/recognition/pointer.py`)
  - 霍夫线变换检测指针
  - 自动表盘中心检测
  - 角度到数值映射

- **Modbus TCP 输出** (`src/retrosight/output/modbus.py`)
  - Modbus TCP Server（伪装 PLC）
  - 多数据类型支持（INT16/FLOAT32等）
  - 寄存器自动分配

- **断网续传缓存** (`src/retrosight/output/buffer.py`)
  - SQLite 持久化存储
  - 优先级队列
  - 自动重试与过期清理

### Phase 3 已实现功能

- **指示灯识别** (`src/retrosight/recognition/light.py`)
  - HSV 颜色检测（红/黄/绿/蓝/白）
  - Andon 三色灯塔状态监控
  - 闪烁检测与频率估算
  - OEE 可用率计算

- **开关/旋钮识别** (`src/retrosight/recognition/switch.py`)
  - 拨动开关 ON/OFF 检测
  - 旋钮档位识别（多档位支持）
  - 按钮/滑动开关状态
  - 模板匹配与位置检测

- **图像增强** (`src/retrosight/preprocessing/enhancement.py`)
  - 自适应对比度增强（CLAHE）
  - 去反光处理（inpainting）
  - 多帧融合降噪
  - 低光照增强

## 许可证

MIT License

## 相关链接

- [文档中心](docs/index.md)
- [产品愿景](docs/roadmap/vision.md)
- [技术挑战](docs/roadmap/challenges.md)

# Open-RetroSight 文档中心

欢迎使用 Open-RetroSight 文档。本项目是一款非侵入式工业边缘AI网关软件，通过计算机视觉将传统"哑设备"的数据数字化。

## 项目状态

🎉 **当前版本：v0.1.0 - MVP 功能完成**

| 阶段 | 状态 |
|-----|------|
| Phase 1: 基础能力 | ✅ 完成 |
| Phase 2: 核心功能 | ✅ 完成 |
| Phase 3: 扩展功能 | ✅ 完成 |
| 集成测试 | ⏳ 进行中 |

## 快速导航

### 入门指南
- [项目介绍](getting-started/introduction.md) - 了解项目背景、核心价值和适用场景

### 开发文档 ⭐ 新增
- [开发进度](development/progress.md) - 模块实现状态与代码统计
- [API 参考](development/api-reference.md) - 核心 API 快速参考
- [变更日志](development/changelog.md) - 版本变更记录

### 应用场景
- [七段数码管/LCD屏幕读取](use-cases/digital-display.md)
- [指针式仪表读取](use-cases/analog-gauge.md)
- [状态指示灯识别](use-cases/indicator-light.md)
- [开关/旋钮位置识别](use-cases/switch-knob.md)

### 技术架构
- [架构总览](architecture/overview.md) - 系统整体架构设计
- [硬件层](architecture/hardware-layer.md) - 硬件选型与配置
- [软件技术栈](architecture/software-stack.md) - 软件组件与技术选型
- [核心算法](architecture/algorithms.md) - 图像处理与识别算法

### 产品规划
- [产品愿景](roadmap/vision.md) - 产品定位与核心价值
- [MVP功能](roadmap/mvp-features.md) - 最小可行产品功能规划
- [技术挑战](roadmap/challenges.md) - 开发难点与解决方案
- [商业化策略](roadmap/commercialization.md) - 生态与盈利模式

## 文档结构

```
docs/
├── getting-started/     # 入门指南
├── development/         # 开发文档 ⭐
│   ├── progress.md      # 开发进度
│   ├── api-reference.md # API 参考
│   └── changelog.md     # 变更日志
├── use-cases/           # 应用场景
├── architecture/        # 技术架构
└── roadmap/             # 产品规划
```

## 快速开始

```bash
# 克隆项目
git clone https://github.com/open-retrosight/open-retrosight.git
cd open-retrosight

# 安装依赖
pip install -r requirements.txt

# 启动 Web 界面
streamlit run src/retrosight/ui/app.py
```

## 相关链接

- [GitHub 仓库](https://github.com/open-retrosight/open-retrosight)
- [问题反馈](https://github.com/open-retrosight/open-retrosight/issues)

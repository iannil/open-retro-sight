# Open-Retro-Sight

> Non-invasive Industrial Edge AI Gateway - Give old machines "digital eyes"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[中文版](README_ZH.md)

## Introduction

Open-RetroSight is a non-invasive industrial edge AI gateway software that digitizes data from traditional "dumb devices" using computer vision technology.

**No disassembly, no downtime, no rewiring** — Achieve digital transformation of legacy equipment with just a few hundred dollars and minutes of deployment.

## Core Capabilities

- **Seven-segment / LCD Display Recognition** - Read temperature, pressure, counters, and other numeric displays
- **Analog Gauge Reading** - Map pointer angles to numerical values
- **Status Indicator Light Detection** - Andon light recognition for OEE calculation
- **Switch / Knob Position Detection** - Identify gear and toggle states

## Quick Start

```bash
# Clone the project
git clone https://github.com/open-retrosight/open-retrosight.git
cd open-retrosight

# Install dependencies
pip install -r requirements.txt

# Or install via pip
pip install -e .
```

## Documentation

For detailed documentation, see [docs/](docs/index.md):

- [Introduction](docs/getting-started/introduction.md) - Background, value proposition, use cases
- [Architecture](docs/architecture/overview.md) - System design and tech stack
- [Use Cases](docs/use-cases/) - Recognition solutions for various devices
- [Roadmap](docs/roadmap/) - Vision, MVP features, commercialization

## Hardware Requirements

- Raspberry Pi Zero 2W / Orange Pi + USB Camera
- Or retired Android phone
- Or ESP32-CAM + Edge Server

## Tech Stack

`Python` `OpenCV` `PaddleOCR` `YOLOv8` `MQTT` `Modbus TCP` `Streamlit`

## Project Status

Current Stage: **MVP v0.1.0 Complete**

### Completed Milestones

- [x] Product planning and design
- [x] Documentation system established
- [x] Project structure initialized
- [x] Phase 1: Core capabilities (video stream, OCR, MQTT, Web UI)
- [x] Phase 2: Key features (pointer recognition, perspective correction, Modbus, offline buffering)
- [x] Phase 3: Extended features (indicator lights, switch recognition, image enhancement)

### Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **v0.1** | MVP Core Features | ✅ Complete |
| **v0.2** | Community Building | Planned |
| **v0.3** | Hardware Kit & Product Polish | Planned |
| **v0.4** | Cloud Platform (SaaS) | Planned |
| **v1.0** | Ecosystem & Marketplace | Planned |

#### Next Steps (v0.2)
- [ ] Build developer community
- [ ] Improve documentation and tutorials
- [ ] Collect user feedback and iterate
- [ ] Add more device recognition templates

#### Future Plans
- **Hardware Kits**: Plug-and-play kits with camera, lighting, and enclosure
- **Advanced Algorithms**: Complex gauge recognition, multi-target detection
- **Cloud Platform**: Multi-device management, data storage, alerting
- **Industry Solutions**: Manufacturing, energy, chemical, logistics

### Phase 1 Implemented Features

- **Video Stream Capture** (`src/retrosight/capture/camera.py`)
  - USB/CSI/RTSP camera support
  - Multi-threaded frame buffering

- **Digital OCR Recognition** (`src/retrosight/recognition/ocr.py`)
  - Seven-segment display recognition (PaddleOCR)
  - Image preprocessing optimization

- **Data Smoothing Filters** (`src/retrosight/preprocessing/filter.py`)
  - Kalman filter, moving average, exponential smoothing
  - Outlier filtering

- **MQTT Data Publishing** (`src/retrosight/output/mqtt.py`)
  - Auto-reconnect, offline caching
  - Structured JSON data

- **Web Configuration Interface** (`src/retrosight/ui/app.py`)
  - Streamlit real-time preview
  - ROI region configuration

### Phase 2 Implemented Features

- **Perspective Transform Correction** (`src/retrosight/preprocessing/transform.py`)
  - Four-point perspective transform
  - Image registration (vibration resistant)
  - Lens distortion correction

- **Pointer Recognition Algorithm** (`src/retrosight/recognition/pointer.py`)
  - Hough line transform for pointer detection
  - Automatic dial center detection
  - Angle to value mapping

- **Modbus TCP Output** (`src/retrosight/output/modbus.py`)
  - Modbus TCP Server (PLC emulation)
  - Multiple data type support (INT16/FLOAT32, etc.)
  - Automatic register allocation

- **Offline Buffer & Retry** (`src/retrosight/output/buffer.py`)
  - SQLite persistent storage
  - Priority queue
  - Auto-retry and expiration cleanup

### Phase 3 Implemented Features

- **Indicator Light Recognition** (`src/retrosight/recognition/light.py`)
  - HSV color detection (red/yellow/green/blue/white)
  - Andon tower light status monitoring
  - Blink detection and frequency estimation
  - OEE availability calculation

- **Switch / Knob Recognition** (`src/retrosight/recognition/switch.py`)
  - Toggle switch ON/OFF detection
  - Rotary knob position recognition (multi-position support)
  - Push button / slide switch states
  - Template matching and position detection

- **Image Enhancement** (`src/retrosight/preprocessing/enhancement.py`)
  - Adaptive contrast enhancement (CLAHE)
  - Glare removal (inpainting)
  - Multi-frame fusion denoising
  - Low-light enhancement

## License

MIT License

## Related Links

- [Documentation Center](docs/index.md)
- [Product Vision](docs/roadmap/vision.md)
- [Technical Challenges](docs/roadmap/challenges.md)

# Open-RetroSight

> Non-invasive Industrial Edge AI Gateway - Give old machines "digital eyes"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**[English](README_EN.md)** | **[中文](README_ZH.md)**

---

Open-RetroSight is a non-invasive industrial edge AI gateway software that digitizes data from traditional "dumb devices" using computer vision technology.

**No disassembly, no downtime, no rewiring** — Achieve digital transformation of legacy equipment with just a few hundred dollars and minutes of deployment.

## Core Capabilities

- **Seven-segment / LCD Display Recognition** - Read temperature, pressure, counters
- **Analog Gauge Reading** - Map pointer angles to numerical values
- **Status Indicator Light Detection** - Andon light recognition for OEE calculation
- **Switch / Knob Position Detection** - Identify gear and toggle states

## Quick Start

```bash
git clone https://github.com/open-retrosight/open-retrosight.git
cd open-retrosight
pip install -r requirements.txt
```

## Tech Stack

`Python` `OpenCV` `PaddleOCR` `YOLOv8` `MQTT` `Modbus TCP` `Streamlit`

## Hardware Requirements

- Raspberry Pi Zero 2W / Orange Pi + USB Camera
- Or retired Android phone
- Or ESP32-CAM + Edge Server

## Documentation

- [Full English Documentation](README_EN.md)
- [完整中文文档](README_ZH.md)
- [Documentation Center](docs/index.md)

## License

MIT License

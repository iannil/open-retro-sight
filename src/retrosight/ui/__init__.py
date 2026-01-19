"""
Web界面模块

基于 Streamlit 的配置界面:
- 实时摄像头预览
- 识别区域框选
- 参数配置
- 校准向导
"""

from retrosight.ui.app import main, AppConfig

__all__ = ["main", "AppConfig"]

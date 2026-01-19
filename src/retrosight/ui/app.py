"""
Streamlit Web 配置界面

功能：
- 实时视频预览
- ROI 区域框选
- OCR 识别结果显示
- MQTT 配置
- 系统状态监控

运行方式:
    streamlit run src/retrosight/ui/app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, asdict
import json
import logging

logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="Open-RetroSight",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@dataclass
class AppConfig:
    """应用配置"""
    # 摄像头配置
    camera_source: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # ROI 配置
    roi_enabled: bool = False
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 200
    roi_height: int = 100

    # OCR 配置
    ocr_enabled: bool = True
    ocr_lang: str = "en"
    ocr_use_gpu: bool = False

    # MQTT 配置
    mqtt_enabled: bool = False
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "retrosight"
    mqtt_username: str = ""
    mqtt_password: str = ""

    # 滤波配置
    filter_enabled: bool = True
    filter_type: str = "kalman"  # kalman, moving_average, exponential
    filter_window_size: int = 5
    filter_alpha: float = 0.3


def init_session_state():
    """初始化会话状态"""
    if "config" not in st.session_state:
        st.session_state.config = AppConfig()

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    if "last_value" not in st.session_state:
        st.session_state.last_value = None

    if "value_history" not in st.session_state:
        st.session_state.value_history = []

    if "mqtt_connected" not in st.session_state:
        st.session_state.mqtt_connected = False


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("⚙️ 配置")

    config = st.session_state.config

    # 摄像头配置
    with st.sidebar.expander("📹 摄像头", expanded=True):
        config.camera_source = st.number_input(
            "设备ID",
            min_value=0,
            max_value=10,
            value=config.camera_source,
            help="USB 摄像头设备编号，通常为 0"
        )

        col1, col2 = st.columns(2)
        with col1:
            config.camera_width = st.selectbox(
                "宽度",
                options=[320, 640, 800, 1280, 1920],
                index=[320, 640, 800, 1280, 1920].index(config.camera_width)
            )
        with col2:
            config.camera_height = st.selectbox(
                "高度",
                options=[240, 480, 600, 720, 1080],
                index=[240, 480, 600, 720, 1080].index(config.camera_height)
            )

        config.camera_fps = st.slider(
            "帧率",
            min_value=1,
            max_value=60,
            value=config.camera_fps
        )

    # ROI 配置
    with st.sidebar.expander("🔲 识别区域 (ROI)", expanded=True):
        config.roi_enabled = st.checkbox(
            "启用 ROI",
            value=config.roi_enabled,
            help="仅识别指定区域内的数字"
        )

        if config.roi_enabled:
            col1, col2 = st.columns(2)
            with col1:
                config.roi_x = st.number_input("X", min_value=0, value=config.roi_x)
                config.roi_width = st.number_input("宽度", min_value=10, value=config.roi_width)
            with col2:
                config.roi_y = st.number_input("Y", min_value=0, value=config.roi_y)
                config.roi_height = st.number_input("高度", min_value=10, value=config.roi_height)

    # OCR 配置
    with st.sidebar.expander("🔤 OCR 识别", expanded=False):
        config.ocr_enabled = st.checkbox(
            "启用 OCR",
            value=config.ocr_enabled
        )

        config.ocr_lang = st.selectbox(
            "语言",
            options=["en", "ch"],
            index=0 if config.ocr_lang == "en" else 1,
            help="en: 英文数字, ch: 中文"
        )

        config.ocr_use_gpu = st.checkbox(
            "使用 GPU",
            value=config.ocr_use_gpu,
            help="需要 CUDA 支持"
        )

    # 滤波配置
    with st.sidebar.expander("📊 数据平滑", expanded=False):
        config.filter_enabled = st.checkbox(
            "启用滤波",
            value=config.filter_enabled
        )

        if config.filter_enabled:
            config.filter_type = st.selectbox(
                "滤波类型",
                options=["kalman", "moving_average", "exponential"],
                format_func=lambda x: {
                    "kalman": "卡尔曼滤波",
                    "moving_average": "滑动平均",
                    "exponential": "指数平滑"
                }[x]
            )

            if config.filter_type == "moving_average":
                config.filter_window_size = st.slider(
                    "窗口大小",
                    min_value=2,
                    max_value=20,
                    value=config.filter_window_size
                )

            if config.filter_type == "exponential":
                config.filter_alpha = st.slider(
                    "平滑系数 (α)",
                    min_value=0.1,
                    max_value=1.0,
                    value=config.filter_alpha,
                    help="越大响应越快"
                )

    # MQTT 配置
    with st.sidebar.expander("📡 MQTT", expanded=False):
        config.mqtt_enabled = st.checkbox(
            "启用 MQTT",
            value=config.mqtt_enabled
        )

        if config.mqtt_enabled:
            config.mqtt_host = st.text_input(
                "Broker 地址",
                value=config.mqtt_host
            )

            config.mqtt_port = st.number_input(
                "端口",
                min_value=1,
                max_value=65535,
                value=config.mqtt_port
            )

            config.mqtt_topic_prefix = st.text_input(
                "主题前缀",
                value=config.mqtt_topic_prefix
            )

            config.mqtt_username = st.text_input(
                "用户名（可选）",
                value=config.mqtt_username
            )

            config.mqtt_password = st.text_input(
                "密码（可选）",
                value=config.mqtt_password,
                type="password"
            )

            if st.button("测试连接"):
                with st.spinner("正在测试连接..."):
                    success, message = test_mqtt_connection(config)
                    if success:
                        st.success(f"✅ {message}")
                        st.session_state.mqtt_connected = True
                    else:
                        st.error(f"❌ {message}")
                        st.session_state.mqtt_connected = False

    # 保存配置按钮
    st.sidebar.divider()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("💾 保存配置", use_container_width=True):
            save_config(config)
            st.success("配置已保存")

    with col2:
        if st.button("📂 加载配置", use_container_width=True):
            loaded = load_config()
            if loaded:
                st.session_state.config = loaded
                st.success("配置已加载")
                st.rerun()


def render_main_content():
    """渲染主内容区域"""
    st.title("👁️ Open-RetroSight")
    st.caption("非侵入式工业边缘AI网关 - 给老机器装上\"数字眼睛\"")

    # 控制按钮
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button(
            "▶️ 启动" if not st.session_state.camera_running else "⏹️ 停止",
            use_container_width=True,
            type="primary" if not st.session_state.camera_running else "secondary"
        ):
            st.session_state.camera_running = not st.session_state.camera_running
            st.rerun()

    with col2:
        if st.button("📸 截图", use_container_width=True):
            st.info("截图功能开发中...")

    with col3:
        if st.button("🔄 重置", use_container_width=True):
            st.session_state.value_history = []
            st.session_state.last_value = None
            st.rerun()

    # 主显示区域
    col_video, col_info = st.columns([2, 1])

    with col_video:
        st.subheader("📺 实时预览")
        video_placeholder = st.empty()

        if st.session_state.camera_running:
            run_camera_loop(video_placeholder)
        else:
            # 显示占位图
            placeholder_img = create_placeholder_image()
            video_placeholder.image(
                placeholder_img,
                caption="点击「启动」开始预览",
                use_container_width=True
            )

    with col_info:
        render_info_panel()


def render_info_panel():
    """渲染信息面板"""
    st.subheader("📊 识别结果")

    # 当前数值
    if st.session_state.last_value is not None:
        st.metric(
            label="当前数值",
            value=f"{st.session_state.last_value:.2f}",
            delta=None
        )
    else:
        st.metric(label="当前数值", value="--")

    # 历史记录
    st.subheader("📈 历史记录")

    if st.session_state.value_history:
        # 显示简单的折线图
        st.line_chart(st.session_state.value_history[-50:])
    else:
        st.info("暂无数据")

    # 系统状态
    st.subheader("🔧 系统状态")

    config = st.session_state.config

    status_items = [
        ("摄像头", "🟢 运行中" if st.session_state.camera_running else "⚪ 停止"),
        ("OCR", "🟢 启用" if config.ocr_enabled else "⚪ 禁用"),
        ("滤波", "🟢 启用" if config.filter_enabled else "⚪ 禁用"),
        ("MQTT", "🟢 已连接" if st.session_state.mqtt_connected else "⚪ 未连接"),
    ]

    for label, status in status_items:
        st.text(f"{label}: {status}")


def run_camera_loop(video_placeholder):
    """运行摄像头循环"""
    config = st.session_state.config

    # 尝试打开摄像头
    cap = cv2.VideoCapture(config.camera_source)

    if not cap.isOpened():
        video_placeholder.error("❌ 无法打开摄像头，请检查设备连接")
        st.session_state.camera_running = False
        return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    cap.set(cv2.CAP_PROP_FPS, config.camera_fps)

    # 初始化滤波器
    filter_obj = create_filter(config)

    # 初始化 OCR（延迟加载）
    ocr_recognizer = None

    try:
        frame_count = 0
        while st.session_state.camera_running:
            ret, frame = cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1

            # 绘制 ROI
            display_frame = frame.copy()
            if config.roi_enabled:
                roi = (config.roi_x, config.roi_y, config.roi_width, config.roi_height)
                cv2.rectangle(
                    display_frame,
                    (roi[0], roi[1]),
                    (roi[0] + roi[2], roi[1] + roi[3]),
                    (0, 255, 0),
                    2
                )

            # 每隔几帧进行 OCR 识别（降低 CPU 占用）
            if config.ocr_enabled and frame_count % 10 == 0:
                try:
                    # 延迟加载 OCR
                    if ocr_recognizer is None:
                        from retrosight.recognition.ocr import OCRRecognizer, OCRConfig
                        ocr_config = OCRConfig(
                            lang=config.ocr_lang,
                            use_gpu=config.ocr_use_gpu,
                            show_log=False
                        )
                        ocr_recognizer = OCRRecognizer(ocr_config)

                    # 识别
                    if config.roi_enabled:
                        roi_frame = frame[
                            config.roi_y:config.roi_y + config.roi_height,
                            config.roi_x:config.roi_x + config.roi_width
                        ]
                        result = ocr_recognizer.recognize(roi_frame)
                    else:
                        result = ocr_recognizer.recognize(frame)

                    # 应用滤波
                    if result.value is not None:
                        if filter_obj and config.filter_enabled:
                            filtered_value = filter_obj.update(result.value)
                        else:
                            filtered_value = result.value

                        st.session_state.last_value = filtered_value
                        st.session_state.value_history.append(filtered_value)

                        # 限制历史记录长度
                        if len(st.session_state.value_history) > 1000:
                            st.session_state.value_history = st.session_state.value_history[-500:]

                        # 在画面上显示数值
                        cv2.putText(
                            display_frame,
                            f"Value: {filtered_value:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                except Exception as e:
                    logger.error(f"OCR 识别失败: {e}")

            # BGR 转 RGB 显示
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # 更新显示
            video_placeholder.image(
                display_frame,
                caption=f"帧: {frame_count}",
                use_container_width=True
            )

            # 控制帧率
            time.sleep(1.0 / config.camera_fps)

    finally:
        cap.release()


def create_filter(config: AppConfig):
    """根据配置创建滤波器"""
    if not config.filter_enabled:
        return None

    try:
        if config.filter_type == "kalman":
            from retrosight.preprocessing.filter import KalmanFilter1D
            return KalmanFilter1D()

        elif config.filter_type == "moving_average":
            from retrosight.preprocessing.filter import MovingAverage
            return MovingAverage(window_size=config.filter_window_size)

        elif config.filter_type == "exponential":
            from retrosight.preprocessing.filter import ExponentialSmoothing
            return ExponentialSmoothing(alpha=config.filter_alpha)

    except ImportError as e:
        logger.error(f"无法加载滤波器: {e}")

    return None


def create_placeholder_image() -> np.ndarray:
    """创建占位图像"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # 深灰色背景

    # 添加文字
    text = "Camera Offline"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    x = (640 - text_size[0]) // 2
    y = (480 + text_size[1]) // 2

    cv2.putText(img, text, (x, y), font, font_scale, (100, 100, 100), thickness)

    return img


def save_config(config: AppConfig, path: str = "config.json"):
    """保存配置到文件"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"保存配置失败: {e}")
        return False


def load_config(path: str = "config.json") -> Optional[AppConfig]:
    """从文件加载配置"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return AppConfig(**data)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return None


def test_mqtt_connection(config: AppConfig) -> Tuple[bool, str]:
    """
    测试 MQTT 连接

    Args:
        config: 应用配置

    Returns:
        (成功标志, 消息)
    """
    try:
        import paho.mqtt.client as mqtt

        # 创建客户端
        client = mqtt.Client(
            client_id=f"retrosight_test_{int(time.time())}",
            protocol=mqtt.MQTTv311
        )

        # 设置认证
        if config.mqtt_username:
            client.username_pw_set(config.mqtt_username, config.mqtt_password)

        # 连接结果
        connection_result = {"success": False, "message": ""}

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                connection_result["success"] = True
                connection_result["message"] = "连接成功"
            else:
                error_messages = {
                    1: "协议版本错误",
                    2: "客户端标识符无效",
                    3: "服务器不可用",
                    4: "用户名或密码错误",
                    5: "未授权"
                }
                connection_result["message"] = error_messages.get(rc, f"未知错误 (rc={rc})")

        client.on_connect = on_connect

        # 尝试连接（5秒超时）
        client.connect(config.mqtt_host, config.mqtt_port, keepalive=5)

        # 等待连接结果
        client.loop_start()
        timeout = 5.0
        start_time = time.time()

        while not connection_result["success"] and not connection_result["message"]:
            if time.time() - start_time > timeout:
                connection_result["message"] = "连接超时"
                break
            time.sleep(0.1)

        client.loop_stop()
        client.disconnect()

        return connection_result["success"], connection_result["message"]

    except ImportError:
        return False, "MQTT 库未安装 (pip install paho-mqtt)"
    except ConnectionRefusedError:
        return False, f"连接被拒绝: {config.mqtt_host}:{config.mqtt_port}"
    except TimeoutError:
        return False, "连接超时"
    except Exception as e:
        return False, f"连接失败: {str(e)}"


def main():
    """主函数"""
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()

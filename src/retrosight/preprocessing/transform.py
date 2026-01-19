"""
透视变换与图像校正模块

功能：
- 四点透视变换：校正摄像头侧拍角度
- 自动边缘检测：自动识别显示区域
- 图像配准：抗振动对齐
- 畸变校正：镜头畸变矫正

用于处理非正视角度拍摄的仪表画面
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """变换配置"""
    auto_detect: bool = False         # 是否自动检测角点
    target_width: int = 200           # 目标宽度
    target_height: int = 100          # 目标高度
    border_margin: int = 5            # 边框边距
    canny_threshold1: int = 50        # Canny 边缘检测阈值1
    canny_threshold2: int = 150       # Canny 边缘检测阈值2


class PerspectiveTransform:
    """
    透视变换器

    用于校正侧拍角度，将倾斜的画面变换为正视图

    使用示例:
    ```python
    # 手动指定四个角点
    transform = PerspectiveTransform()
    src_points = [(10, 10), (200, 15), (195, 100), (5, 95)]
    transform.set_source_points(src_points)

    # 变换图像
    corrected = transform.apply(image)

    # 或使用自动检测
    transform = PerspectiveTransform(TransformConfig(auto_detect=True))
    corrected = transform.apply(image)
    ```
    """

    def __init__(self, config: Optional[TransformConfig] = None):
        """
        初始化透视变换器

        Args:
            config: 变换配置
        """
        self.config = config or TransformConfig()
        self._src_points: Optional[np.ndarray] = None
        self._dst_points: Optional[np.ndarray] = None
        self._matrix: Optional[np.ndarray] = None
        self._inverse_matrix: Optional[np.ndarray] = None

    def set_source_points(
        self,
        points: List[Tuple[float, float]],
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        设置源图像的四个角点

        Args:
            points: 四个角点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                    顺序：左上、右上、右下、左下
            target_size: 目标尺寸 (width, height)，默认使用配置值
        """
        if len(points) != 4:
            raise ValueError("必须提供4个角点")

        self._src_points = np.array(points, dtype=np.float32)

        # 设置目标点（矩形）
        w = target_size[0] if target_size else self.config.target_width
        h = target_size[1] if target_size else self.config.target_height

        self._dst_points = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        # 计算变换矩阵
        self._matrix = cv2.getPerspectiveTransform(
            self._src_points,
            self._dst_points
        )
        self._inverse_matrix = cv2.getPerspectiveTransform(
            self._dst_points,
            self._src_points
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        应用透视变换

        Args:
            image: 输入图像

        Returns:
            校正后的图像
        """
        if self.config.auto_detect and self._matrix is None:
            # 自动检测角点
            points = self._detect_corners(image)
            if points is not None:
                self.set_source_points(points)

        if self._matrix is None:
            logger.warning("未设置变换矩阵，返回原图")
            return image

        w = self.config.target_width
        h = self.config.target_height

        return cv2.warpPerspective(image, self._matrix, (w, h))

    def apply_inverse(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        应用逆透视变换（将校正后的图像变换回原视角）

        Args:
            image: 校正后的图像
            target_size: 目标尺寸 (width, height)

        Returns:
            变换后的图像
        """
        if self._inverse_matrix is None:
            return image

        return cv2.warpPerspective(image, self._inverse_matrix, target_size)

    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        变换单个点坐标

        Args:
            point: 源图像中的点 (x, y)

        Returns:
            变换后的点坐标
        """
        if self._matrix is None:
            return point

        src = np.array([[point]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self._matrix)
        return (float(dst[0][0][0]), float(dst[0][0][1]))

    def transform_points(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        变换多个点坐标

        Args:
            points: 点坐标列表

        Returns:
            变换后的点坐标列表
        """
        return [self.transform_point(p) for p in points]

    def _detect_corners(self, image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        自动检测图像中的四边形角点

        Args:
            image: 输入图像

        Returns:
            四个角点坐标，检测失败返回 None
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(
            blurred,
            self.config.canny_threshold1,
            self.config.canny_threshold2
        )

        # 膨胀边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 多边形近似
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 检查是否为四边形
        if len(approx) != 4:
            return None

        # 排序角点：左上、右上、右下、左下
        points = approx.reshape(4, 2)
        return self._order_points(points)

    def _order_points(self, points: np.ndarray) -> List[Tuple[float, float]]:
        """
        按顺序排列四个角点

        Args:
            points: 四个点的坐标

        Returns:
            排序后的角点列表 [左上, 右上, 右下, 左下]
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        # 左上角的 x+y 最小，右下角的 x+y 最大
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        # 右上角的 y-x 最小，左下角的 y-x 最大
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        return [(float(p[0]), float(p[1])) for p in rect]

    def reset(self):
        """重置变换器"""
        self._src_points = None
        self._dst_points = None
        self._matrix = None
        self._inverse_matrix = None

    @property
    def is_configured(self) -> bool:
        """是否已配置变换矩阵"""
        return self._matrix is not None


class ImageRegistration:
    """
    图像配准器

    用于抗振动对齐，将抖动的图像配准到参考帧

    使用示例:
    ```python
    reg = ImageRegistration()
    reg.set_reference(reference_image)

    for frame in video_frames:
        aligned = reg.align(frame)
    ```
    """

    def __init__(self, method: str = "orb"):
        """
        初始化图像配准器

        Args:
            method: 特征检测方法 ("orb", "sift", "akaze")
        """
        self.method = method
        self._reference: Optional[np.ndarray] = None
        self._reference_keypoints = None
        self._reference_descriptors = None
        self._detector = self._create_detector(method)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _create_detector(self, method: str):
        """创建特征检测器"""
        if method == "orb":
            return cv2.ORB_create(nfeatures=500)
        elif method == "akaze":
            return cv2.AKAZE_create()
        elif method == "sift":
            return cv2.SIFT_create()
        else:
            raise ValueError(f"不支持的方法: {method}")

    def set_reference(self, image: np.ndarray):
        """
        设置参考图像

        Args:
            image: 参考图像
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        self._reference = gray

        # 检测特征点
        self._reference_keypoints, self._reference_descriptors = \
            self._detector.detectAndCompute(gray, None)

    def align(self, image: np.ndarray) -> np.ndarray:
        """
        将图像配准到参考帧

        Args:
            image: 待配准图像

        Returns:
            配准后的图像
        """
        if self._reference is None:
            return image

        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 检测特征点
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)

        if descriptors is None or self._reference_descriptors is None:
            return image

        # 特征匹配
        try:
            matches = self._matcher.match(descriptors, self._reference_descriptors)
        except cv2.error:
            return image

        if len(matches) < 4:
            return image

        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)

        # 取前 N 个最佳匹配
        good_matches = matches[:min(50, len(matches))]

        # 提取匹配点
        src_pts = np.float32([
            keypoints[m.queryIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            self._reference_keypoints[m.trainIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)

        # 计算变换矩阵
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if matrix is None:
            return image

        # 应用变换
        h, w = self._reference.shape[:2]
        aligned = cv2.warpPerspective(image, matrix, (w, h))

        return aligned

    def reset(self):
        """重置配准器"""
        self._reference = None
        self._reference_keypoints = None
        self._reference_descriptors = None


class LensDistortionCorrector:
    """
    镜头畸变校正器

    校正广角镜头的桶形/枕形畸变

    使用示例:
    ```python
    corrector = LensDistortionCorrector()

    # 使用棋盘格标定
    corrector.calibrate_with_chessboard(calibration_images)

    # 校正图像
    undistorted = corrector.undistort(image)
    ```
    """

    def __init__(self):
        """初始化畸变校正器"""
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        self._new_camera_matrix: Optional[np.ndarray] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None

    def calibrate_with_chessboard(
        self,
        images: List[np.ndarray],
        pattern_size: Tuple[int, int] = (9, 6),
        square_size: float = 1.0
    ) -> bool:
        """
        使用棋盘格图像进行标定

        Args:
            images: 棋盘格图像列表
            pattern_size: 棋盘格内角点数 (列, 行)
            square_size: 棋盘格方格大小（任意单位）

        Returns:
            是否标定成功
        """
        # 准备对象点
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        obj_points = []  # 3D 点
        img_points = []  # 2D 点

        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                obj_points.append(objp)

                # 亚像素精确化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_refined)

        if len(obj_points) < 3:
            logger.warning("有效标定图像不足")
            return False

        # 标定
        h, w = images[0].shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )

        if not ret:
            return False

        self._camera_matrix = mtx
        self._dist_coeffs = dist

        # 计算最优新相机矩阵
        self._new_camera_matrix, self._roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h)
        )

        logger.info("相机标定成功")
        return True

    def set_coefficients(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ):
        """
        直接设置标定参数

        Args:
            camera_matrix: 相机内参矩阵 (3x3)
            dist_coeffs: 畸变系数
        """
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs

    def undistort(self, image: np.ndarray, crop: bool = True) -> np.ndarray:
        """
        校正镜头畸变

        Args:
            image: 输入图像
            crop: 是否裁剪到有效区域

        Returns:
            校正后的图像
        """
        if self._camera_matrix is None or self._dist_coeffs is None:
            return image

        # 使用最优相机矩阵或原矩阵
        new_mtx = self._new_camera_matrix if self._new_camera_matrix is not None \
            else self._camera_matrix

        # 校正畸变
        undistorted = cv2.undistort(
            image,
            self._camera_matrix,
            self._dist_coeffs,
            None,
            new_mtx
        )

        # 裁剪到有效区域
        if crop and self._roi is not None:
            x, y, w, h = self._roi
            undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    @property
    def is_calibrated(self) -> bool:
        """是否已标定"""
        return self._camera_matrix is not None

    def save_calibration(self, filepath: str):
        """保存标定参数"""
        if not self.is_calibrated:
            raise ValueError("未标定")

        np.savez(
            filepath,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            new_camera_matrix=self._new_camera_matrix,
            roi=np.array(self._roi) if self._roi else None
        )

    def load_calibration(self, filepath: str):
        """加载标定参数"""
        data = np.load(filepath)
        self._camera_matrix = data["camera_matrix"]
        self._dist_coeffs = data["dist_coeffs"]
        self._new_camera_matrix = data.get("new_camera_matrix")
        roi = data.get("roi")
        self._roi = tuple(roi) if roi is not None else None


def four_point_transform(
    image: np.ndarray,
    points: List[Tuple[float, float]],
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    便捷函数：四点透视变换

    Args:
        image: 输入图像
        points: 四个角点 [左上, 右上, 右下, 左下]
        target_size: 目标尺寸 (width, height)

    Returns:
        变换后的图像
    """
    transform = PerspectiveTransform()
    if target_size:
        transform.config.target_width = target_size[0]
        transform.config.target_height = target_size[1]
    transform.set_source_points(points)
    return transform.apply(image)


def auto_perspective_correct(image: np.ndarray) -> np.ndarray:
    """
    便捷函数：自动透视校正

    Args:
        image: 输入图像

    Returns:
        校正后的图像
    """
    config = TransformConfig(auto_detect=True)
    transform = PerspectiveTransform(config)
    return transform.apply(image)

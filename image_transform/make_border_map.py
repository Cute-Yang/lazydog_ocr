"""
 阈值图标签
- MakeBorderMap:
    shrink_ratio: 0.4
    thresh_min: 0.3
    thresh_max: 0.7
 概率图标签
- MakeShrinkMap:
    shrink_ratio: 0.4
    min_text_size: 8
"""

import traceback
from typing import List, Tuple, Union

import cv2
import numpy as np
import pyclipper
from numpy.typing import NDArray
from shapely.geometry import Polygon


class MakeBorderMap(object):
    def __init__(self, shrink_ratio: float = 0.4, minimum_thresh: float = 0.3, maximum_thresh: float = 0.7):
        """
        Args:
            shrink_ratio:float,收缩因子,D = A * (1 -r * r) / L
                其中A是四边形面积,L是四边形的周长
        """
        self.shrink_ratio = shrink_ratio
        self.minimum_thresh = minimum_thresh
        self.maximum_thresh = maximum_thresh

    def __call__(self, image: NDArray = None, text_polygons=None) -> Tuple[NDArray]:
        """
        Args:
            image:NDArray,the orginal image
            text_polygons:the list of polygon (text)
        """
        assert image.ndim == 3, "the image should be 3-D array,but get {}-D array".format(image.ndim)
        h, w, _ = image.shape
        canvas = np.zeros(
            shape=(h, w),
            dtype=np.float32
        )
        mask = np.zeros(
            shape=(h, w),
            dtype=np.float32
        )
        for i in range(len(text_polygons)):
            print("process polygon {}".format(text_polygons[i]))
            self.draw_border_map(text_polygons[i], canvas=canvas, mask=mask)
        return canvas, mask

    def draw_border_map(self, polygon: Union[List[List[int]], NDArray], canvas: NDArray = None, mask: NDArray = None):
        """
        Args
            polygon:文本框的四个点坐标，按照顺时针进行排列
            canvas:np.ndarray,画布
            mask:np.ndarray,mask..
        """
        if not isinstance(polygon, np.ndarray):
            try:
                polygon = np.array(polygon, dtype=np.int32)
            except:
                traceback.print_exc()
                exit(-1)

        assert polygon.ndim == 2, "the polygon should be 2-D array,but get {}=D array!".format(polygon.ndim)
        assert polygon.shape[1] == 2, "the polygon's point should contain 2 element,but get {}".format(polygon.shape[1])

        # 四个点组成的面
        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0:
            return

        # 计算偏移量
        distance = polygon_shape.area * (1 - self.shrink_ratio * self.shrink_ratio) / polygon_shape.length

        # 将点的结构转换成list
        points = [tuple(p) for p in polygon]

        # 完全就是对c++的封装
        # 对多边形进行扩张
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # 取扩张的最外层
        padded_polygon: NDArray = np.array(padding.Execute(distance))[0]
        # 将多边形的mask置为1
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
        # 得到概率图和阈值图之间的区域
        # 让其实位置变为0
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()

        # 所以这里出来的一定是整数咯
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        # 产生自然增长序列,可以为什么不用range?
        # 做一个笛卡尔交叉?
        xs = np.broadcast_to(
            np.linspace(0, width - 1, width).reshape(1, width), (height, width)
        )

        ys = np.broadcast_to(
            np.linspace(0, height - 1, height).reshape(height, 1), (height, width)
        )
        # 有几个点，就要计算几次
        distance_map = np.zeros(
            shape=(polygon.shape[0], height, width),
            dtype=np.float32
        )
        for i in range(polygon.shape[0]):
            # 为了保证 p0p1 p1p2 .... pi-2p0连接
            # 如果 是四边形
            # 那么计算的点就是 p0p1 p1p2 p2p3 p3p0,完成闭环操作
            j = (i + 1) if i < (polygon.shape[0] - 1) else 0
            absolute_distance = self._distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0.0, 1.0)

        # 计算出区域内到每条边的距离，然后取到最近一条边的距离
        distance_map = distance_map.min(axis=0)
        print(distance_map.shape)

        # 防止越界
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        print("xmin:{} ymin:{} xmax:{} ymax:{}".format(xmin_valid, ymin_valid, xmax_valid, ymax_valid))

        # 将上述的距离 1-x
        # 这是为了防止出现像素重叠吗？谁离得近就算谁的
        # 这个好理解
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,
                             xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1]
        )
        

    # 计算多边形区域的点到原始框边的点
    def _distance(self, xs: NDArray = None, ys: NDArray = None, point_1: List[int] = None, point_2: List[int] = None):
        """
        Args:
            xs:ndarray,the value of all x coor of this area
            ys:ndarraym,the value of all y coor of this area
            tips:the coor of this area is combined by s[i,j] = (xs[i],ys[i])
            point_1:the x,y of first point
            point2_2:the x,y of second point!
        """
        assert len(point_1) == 2, "the element sizeof point must be 2,but get {}".format(len(point_1))
        assert len(point_2) == 2, "the element sizeof point must be 2,but get {}".format(len(point_2))
        # 计算区域内的点和第一个点的距离
        square_distance_from_point_1: NDArray = np.square(xs - float(point_1[0])) + np.square(ys - float(point_1[1]))
        # 计算区域内的点和第二个点的距离
        square_distance_from_point_2: NDArray = np.square(xs - float(point_2[0])) + np.square(ys - float(point_2[1]))

        # 计算给定线段的边长
        square_line_distance: float = np.square(float(point_1[0]) - float(point_2[0])) + np.square(float(point_1[1]) - float(point_2[1]))

        # 根据余弦定理计算cos值，但是这里计算的是
        # 这里计算的是给定线段作为对边的角的余弦值
        cosin = (square_line_distance - square_distance_from_point_1 - square_distance_from_point_2) / \
           (2 * np.sqrt(square_distance_from_point_1 * square_distance_from_point_2))

        square_sin = 1 - np.square(cosin)
        # 避免出现nan值
        square_sin = np.nan_to_num(square_sin)

        # 利用面积公式计算高度
        h = np.sqrt(square_distance_from_point_1 * square_distance_from_point_2 * square_sin / square_line_distance)

        # 但是这一步我就不理解了
        # 这一步为什么要挑选这些呢?
        ne_mask = (cosin < 0)
        h[ne_mask] = np.sqrt(np.fmin(square_distance_from_point_1, square_distance_from_point_2))[ne_mask]
        return h

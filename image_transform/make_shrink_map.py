"""
make the shrink ratio...
"""

from typing import List, Tuple

import cv2
import numpy as np
# 使用这个库来进行多边形的膨胀和收缩
import pyclipper
from numpy.typing import NDArray
from shapely.geometry import Polygon


class MakeShrinkMap(object):
    def __init__(self, min_text_size: 8, shrink_ratio: 0.4, **kwargs):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, image: NDArray, text_polygons=None, ignore_tags: List[bool] = None) -> Tuple:
        h, w, _ = image.shape
        if ignore_tags is None:
            ignore_tags = [False for _ in range(len(text_polygons))]
        if not isinstance(text_polygons,np.ndarray):
            text_polygons = np.array(text_polygons,dtype=np.float32)
        text_polygons, ignore_tags = self.validate_polygons(
            polygons=text_polygons,
            ignore_tags=ignore_tags,
            h=h,
            w=w
        )
        print(ignore_tags)

        gt = np.zeros(
            shape=(h, w),
            dtype=np.float32
        )
        mask = np.ones(
            shape=(h, w),
            dtype=np.float32
        )
        for i in range(len(text_polygons)):
            polygon = text_polygons[i]
            # 计算能够涵盖这个多边形的矩形区域
            xmin, xmax = np.min(polygon[:, 0]), np.max(polygon[:, 0])
            ymin, ymax = np.min(polygon[:, 1]), np.max(polygon[:, 1])
            rect_h = ymax - ymin
            rect_w = xmax - xmin
            # 如果这个框太小了
            if ignore_tags[i] or min(rect_h, rect_w) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True

            else:
                polygon_shape = Polygon(polygon)
                subject = [tuple(p) for p in polygon]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = []
                shrink_ratios = np.arange(self.shrink_ratio, 1, self.shrink_ratio)
                np.append(shrink_ratios, 1.0)
                for ratio in shrink_ratios:
                    distance = polygon_shape.area * (1 - ratio * ratio) / polygon_shape.length
                    shrinked = padding.Execute(-distance)

                    if len(shrinked) == 1:
                        break

                if len(shrinked) == 0:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue

                shrink = np.array(shrinked[0]).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt, [shrink], 1)

        return gt,mask
    # 验证文本多边形的有效性
    def validate_polygons(self, polygons, ignore_tags: List[bool], h: int, w: int):
        """
        Args:
            polygons:NDArray,has shape (batch_size,num_points,2),but usually,the num_points is 2
            ignore_tags:if False,just ignore it
            h:int,the height of image
            w:int,the width of image
        """
        if len(polygons) == 0:
            return polygons, ignore_tags

        # clip the polygon coordinates...
        for i in range(len(polygons)):
            # this is all x coordinates
            polygons[i, :, 0] = np.clip(polygons[i, :, 0], 0, w - 1)
            # this is all the y coordinates

            polygons[i, :, 1] = np.clip(polygons[i, :, 1], 0, h - 1)

        for i in range(len(polygons)):
            # make sure the area is valid!
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            # 这个可能是点的位置不对
            if area > 0:
                # 从顺时针变成逆时针?
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    # 可以使用Polygon来完成呀，不过也可以手动计算
    def polygon_area(self, polygon) -> float:
        area = 0.0
        p1 = polygon[-1]
        # 同样也是为了我
        # 这里刚好是沿着逆时针方向进行计算#
        # 百度的那里取余操作是沿着顺时针方向进行计算
        for p in polygon:
            # 就是计算两个点的组成的行列式，或者叫做向量的叉乘
            area += p[0] * p1[1] - p[1] * p1[0]
            # update the next point
            p1 = p
        return area * 0.5

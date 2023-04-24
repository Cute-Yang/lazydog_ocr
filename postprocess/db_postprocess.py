"""
postprocess for the shrink_maps
"""
from typing import List, Tuple

import cv2
import numpy as np
import pyclipper
# 用来计算多边形的面积和周长
from numpy.typing import NDArray
from shapely.geometry import Polygon


class DBPostPorcess(object):
    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates: int = 1000,
                 unclip_ratio: float = 2.0,
                 use_dilation=False,
                 score_mod: str = "fast",
                 box_type="quad"):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        # 扩张系数,之前制作的时候 使用了0,4的收缩系数
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.use_dilation = use_dilation
        self.score_mod = score_mod
        self.box_type = box_type

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def polygons_from_bitmap(self, shrink_map: NDArray, bitmap: NDArray = None, dst_width: int = None, dst_height: int = None):
        """
        Args:
            shrink_map:ndarray,the output of db model 2-D array
            bitmap:ndarray,if the value > thresh,set -> 1,else 0
            dst_width:int,the width of image
            dst_height:int,the height of image
        """
        h, w = bitmap.shape
        boxes = []
        scores = []
        contours, _ = cv2.findContours((bitmap*255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            apporx = cv2.approxPolyDP(contour, epsilon, True)
            points = apporx.reshape((-1, 2))
            if len(points) < 4:
                continue
            score = self.box_score_fast(shrink_map, points)
            if score < self.box_thresh:
                continue
            box = self.unclip(points, self.unclip_ratio)
            box = box.reshape((-1, 2))
            _, ssid = self.get_mini_boxes(box.reshape(-1, 1, 2))
            if ssid < self.min_size + 2:
                continue
            box = np.array(box)
            # 按照比例放大
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dst_width), 0, dst_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dst_height), 0, dst_height)
            boxes.append(box.tolist())
            scores.append(score)
            return boxes, scores

    def boxes_from_bitmap(self, shrink_map: NDArray, bitmap: NDArray, dst_width: int, dst_height: int) -> Tuple[List, List]:
        print(bitmap.shape)
        h, w = bitmap.shape
        outs = cv2.findContours((bitmap*255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            contours = outs[1]
        elif len(outs) == 2:
            contours = outs[0]
        num_contours = min(self.max_candidates, len(contours))
        boxes, scores = [], []
        for i in range(num_contours):
            contour = contours[i]
            points, small_side = self.get_mini_boxes(contour=contour)
            if small_side < self.min_size:
                continue
            points = np.array(points)
            if self.score_mod == "fast":
                score = self.box_score_fast(shrink_map, points)
            else:
                score = self.box_score_slow(shrink_map, points)
            if score < self.box_thresh:
                continue

            box = self.unclip(points).reshape((-1, 1, 2))
            box, small_side = self.get_mini_boxes(box)
            if small_side < self.min_size + 2:
                continue
            box = np.array(box)

            # clip
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dst_width), 0, dst_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dst_height), 0, dst_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def box_score_fast(self, shrink_map: NDArray, box: NDArray = None) -> float:
        h, w = shrink_map.shape
        box = np.copy(box)
        # 计算坐标像素
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.ceil(box[:, 1].min()).astype(np.int32), 0, h-1)
        xmax = np.clip(np.floor(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymax = np.clip(np.floor(box[:, 1].max()).astype(np.int32), 0, h-1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape((1, -1, 2)).astype(np.int32), 1)
        return cv2.mean(shrink_map[ymin:ymax+1, xmin:xmax+1], mask)[0]

    def box_score_slow(self, shrink_map: NDArray, box) -> float:
        h, w = shrink_map.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(shrink_map[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box):
        polygon = Polygon(box)
        # 使用行列式的值来进行计算
        distance = polygon.area * self.unclip_ratio / polygon.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        # 最小外接矩形,带旋转
        bounding_box = cv2.minAreaRect(contour)
        # 先根据x coor进行排序 sort排序默认是升序
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        # 再顺时针旋转
        i1, i2, i3, i4 = 0, 1, 2, 3
        # 上下交换,交换后满足顺时针排序
        if points[1][1] > points[0][1]:
            i1 = 0
            i4 = 1
        else:
            i1 = 1
            i4 = 0
        if points[3][1] > points[2][1]:
            i2 = 2
            i3 = 3
        else:
            i2 = 3
            i3 = 2
        box = [points[i1], points[i2], points[i3], points[i4]]
        # 返回短边长度
        ssid = min(bounding_box[1])  # bounding box是一个c++类,center,size,angle,... size 是一个sturct
        return box, ssid

    def __call__(self, batch_shrink_map: NDArray, image_shape_info_list: List[int]) -> Tuple:
        """
        Args:
            batch_shrink_map:ndarray from model output
            image_info_list:list contain the h and w of image
        """
        # strip the channle dim
        batch_shrink_map = np.squeeze(batch_shrink_map,axis=1)
        batch_size = len(batch_shrink_map)
        batch_segmentation = batch_shrink_map > self.thresh
        result = []
        for i in range(batch_size):
            src_h, src_w = image_shape_info_list[i]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(np.arange(batch_segmentation[i]).astype(np.uint8), self.dilation_kernel)
            else:
                mask = batch_segmentation[i]
            if self.box_type == "poly":
                boxes, scores = self.polygons_from_bitmap(batch_shrink_map[i], mask, src_w, src_h)
            else:
                boxes, scores = self.boxes_from_bitmap(batch_shrink_map[i], mask, src_w, src_h)

            result.append(
                {
                    "points": boxes,
                    "scores": scores
                }
            )
        return result

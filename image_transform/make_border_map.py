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
from typing import List, Union

import cv2
import numpy as np
import pyclipper
from numpy.typing import NDArray
from shapely.geometry import Polygon


class MakeBorderMap(object):
    def __init__(self,shrink_ratio:float=0.4,minimum_thresh:float=0.3,maximum_thresh:float=0.7):
        """
        Args:
            shrink_ratio:float,收缩因子,D = A * (1 -r * r) / L
                其中A是四边形面积,L是四边形的周长
        """
        self.shrink_ratio = shrink_ratio
        self.minimum_thresh = minimum_thresh
        self.maximum_thresh = maximum_thresh
        


    def draw_border_map(self,polygon:Union[List[List[int]],NDArray],canvas:NDArray=None,mask:NDArray=None):
        """
        Args
            polygon:文本框的四个点坐标，按照顺时针进行排列
            canvas:np.ndarray,画布
            mask:np.ndarray,mask..
        """
        if not isinstance(polygon,NDArray):
            try:
                polygon = np.ndarray(polygon,dtype=np.int32)
            except:
                traceback.print_exc()
                exit(-1)
        
        assert polygon.ndim == 2,"the polygon should be 2-D array,but get {}=D array!".format(polygon.ndim)
        assert polygon.shape[1] == 2,"the polygon's point should contain 2 element,but get {}".format(polygon.shape[1])
        
        # 四个点组成的面
        polygon_shape = Polygon(polygon)
        if polygon_shape.area <=0:
            return
        
        # 计算偏移量
        distance = polygon_shape.area * ( 1- self.shrink_ratio * self.shrink_ratio) / polygon_shape.area
        
        # 将点的结构转换成list
        points = [tuple(p) for p in polygon]
        
        # 完全就是对c++的封装
        #对多边形进行扩张
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon:NDArray = np.array(padding.Execute(distance))[0]
        cv2.fillPoly(mask,[padded_polygon.astype(np.int32)],0)
        #得到概率图和阈值图之间的区域
        #让其实位置变为0
        xmin = padded_polygon[:,0].min()
        xmax = padded_polygon[:,0].max()
        ymin = padded_polygon[:,1].min()
        ymax = padded_polygon[:,1].max()
        
        #所以这里出来的一定是整数咯
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        
        polygon[:,0] = polygon[:,0] - xmin
        polygon[:,1] = polygon[:,1] - ymax
        
        # 产生自然增长序列,可以为什么不用range?
        #做一个笛卡尔交叉?
        xs = np.broadcast_to(
            np.linspace(0,width -1,width).reshape(1,width),(height,width)
        )

        ys =  np.broadcast_to(
            np.linspace(0,height -1,height).reshape(height,1),(height,width)
        )
        distance_map = np.zeros(
            shape=(polygon.shape[0],height,width),
            dtype=np.float32
        )
        for i in rran
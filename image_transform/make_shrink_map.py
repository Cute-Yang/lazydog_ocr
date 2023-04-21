"""
make the shrink ratio...
"""

from typing import List, Tuple

import cv2
import numpy as np
import pyclipper
from numpy.typing import NDArray
from shapely.geometry import Polygon


class MakeShrinkMap(object):
    def __init__(self,min_text_size:8,shrink_ratio:0.4,**kwargs):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self,image:NDArray,text_polygons=None,ignore_tags:List[bool]=None) -> Tuple:
        h,w,_ = image.shape
        if ignore_tags is None:
            ignore_tags = [False for _ in range(text_polygons)]
    

    # 验证文本多边形的有效性
    def validate_polygons(self,polygons,h,w):
        if len(polygons) == 0:
            return [],[]
        
        
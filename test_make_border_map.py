import json

import cv2
import numpy as np

from image_transform.make_border_map import MakeBorderMap


def get_image_and_polygon_from_file(src_file: str = None):
    with open(src_file, "r", encoding="utf-8") as f:
        # 仅仅读取第一行的测试内容作为测试
        line = f.readline()
        # 默认以\t作为分隔符
    line_split = line.strip().split("\t")
    assert len(line_split) == 2, "the size of label line should be 2,but get {}".format(len(line_split))
    image_path, text_polygon_info = line_split
    text_polygon_info_dict = json.loads(text_polygon_info)
    text_polygons = [item["points"] for item in text_polygon_info_dict]
    image = cv2.imread(image_path)
    return image, text_polygons


if __name__ == "__main__":
    src_file = "real_world_text_images/Label.txt"
    image, text_polygons = get_image_and_polygon_from_file(src_file=src_file)
    make_border_map_worker = MakeBorderMap()
    threshold_map, mask = make_border_map_worker(image=image, text_polygons=text_polygons)
    cv2.imwrite("threshold_map.png",(threshold_map * 255.0).astype(np.uint8))
    cv2.imwrite("mask.png",(mask * 255.0).astype(np.uint8))

import json

import cv2
import numpy as np

from image_transform.make_shrink_map import MakeShrinkMap

if __name__ == "__main__":
    src_file = "real_world_text_images/Label.txt"
    with open(src_file, "r", encoding="utf-8") as f:
        label_line = f.readline()
    line_split = label_line.strip().split("\t")
    assert len(line_split) == 2, "the label line should have len 2,but get {}".format(len(line_split))
    image_path, polygon_info_list = line_split
    image = cv2.imread(image_path)
    polygon_info_list = json.loads(polygon_info_list)
    text_polygons = [item["points"] for item in polygon_info_list]
    make_shrink_map_worker = MakeShrinkMap(
        min_text_size=8,
        shrink_ratio=0.4
    )
    gt, mask = make_shrink_map_worker(
        image=image,
        text_polygons=text_polygons
    )
    gt = (gt * 255.0).astype(np.uint8)
    mask = (mask * 255.0).astype(np.uint8)

    cv2.imwrite("shrink_map.png", gt)
    cv2.imwrite("shrink_mask.png", mask)

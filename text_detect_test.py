import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

from backbones.mobilenetV3 import MobileNetV3
from head.db_head import DBHead
from neck.rsefpn import RSEFPN
from postprocess.db_postprocess import DBPostPorcess


def compute_resize_info(h: int, w: int, max_side_len: int = 960):
    max_side = max(h, w)
    scale = 1.0
    if max_side > max_side_len:
        scale = max_side_len / max_side
    h = h * scale
    w = w * scale
    h = int(round(h / 32.0)) * 32
    w = int(round(w / 32.0)) * 32
    return h, w


def image_preprocess(image: NDArray = None, add_batch_size_dim: bool = True) -> NDArray:
    """
    Args:
        image:the input image,bgr channel
        add_batch_size_dim:bool,if True,we will expadn_dim at 0!
    """
    # mean and std
    h, w, _ = image.shape
    new_h, new_w = compute_resize_info(h=h, w=w)
    image = cv2.resize(image, (new_w, new_h))
    normalize_means = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    normalize_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = image.astype(dtype=np.float32) / 255.0
    image = (image - normalize_means) / normalize_stds
    # transpose
    image = np.transpose(image, axes=(2, 0, 1))
    if add_batch_size_dim:
        image = np.expand_dims(image, axis=0)
    return image


class DBNet(nn.Module):
    def __init__(self) -> None:
        super(DBNet, self).__init__()
        self.backbone = MobileNetV3(disable_se=True)
        self.neck = RSEFPN(
            feature_map_channels_list=self.backbone.get_feature_map_channels_list(),
            out_channels=96,
            shortcut=True
        )
        self.head = DBHead(in_channels=96, k=50)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        # for feat in feats:
        #     print(feat.shape)
        output = self.neck(feats)
        output = self.head(output)
        return output


if __name__ == "__main__":
    db_net = DBNet()
    checkpoint = "models/db_net.pth"
    state_dict = torch.load(checkpoint)
    db_net.load_state_dict(state_dict)
    db_net.eval()
    # image_path = "real_world_text_images/street.png"
    image_path = "real_world_text_images/sky.jpg"
    image = cv2.imread(image_path)
    processed_image = image_preprocess(image=image)
    image_shape_info_list = []
    h, w, _ = image.shape
    db_posrtporcess_worker = DBPostPorcess()
    image_shape_info_list.append(
        (h, w)
    )
    with torch.no_grad():
        output = db_net(torch.tensor(processed_image, dtype=torch.float32)).numpy()

    result = db_posrtporcess_worker(
        batch_shrink_map=output,
        image_shape_info_list=image_shape_info_list
    )
    points = result[0]["points"]
    for p in points:
        image = cv2.polylines(image,[p.astype(np.int32)],1,(0,255,0),2)
    
    cv2.imwrite("text_detect_result.png",image)
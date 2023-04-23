from typing import List

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from backbones.mobilenetV3 import MobileNetV3
from experimental.state_dict_keys_replacement import (
    absolutely_dbnet_state_dict_keys_replacement,
    partial_dbnet_state_dict_keys_replacement)
from head.db_head import DBHead
from neck.rsefpn import RSEFPN


def compute_resize_info(h:int,w:int,max_side_len:int=960):
    max_side = max(h,w)
    scale = 1.0
    if max_side > max_side_len:
        scale = max_side_len / max_side
    h = h * scale
    w = w * scale
    h = int(round(h / 32.0)) * 32
    w = int(round(w / 32.0)) * 32
    return h,w


def image_preprocess(image:NDArray=None,add_batch_size_dim:bool=True) -> NDArray:
    """
    Args:
        image:the input image,bgr channel
        add_batch_size_dim:bool,if True,we will expadn_dim at 0!
    """
    # mean and std
    h,w,_ = image.shape
    new_h,new_w = compute_resize_info(h=h,w=w)
    image = cv2.resize(image,(new_w,new_h))
    normalize_means = np.array([0.485,0.456,0.406],dtype=np.float32).reshape(1,1,3)
    normalize_stds = np.array([0.229,0.224,0.225],dtype=np.float32).reshape(1,1,3)
    image = image.astype(dtype=np.float32) / 255.0
    image = (image - normalize_means) / normalize_stds
    # transpose
    image = np.transpose(image,axes=(2,0,1))
    if add_batch_size_dim:
        image = np.expand_dims(image,axis=0)
    return image


def get_keys_dict(weight_keys:List[str]):
    replacement_list = partial_dbnet_state_dict_keys_replacement()
    new_weight_keys = {}
    for key in weight_keys:
        new_key = key
        for l,r in replacement_list:
            new_key = new_key.replace(l,r)
        new_weight_keys[key,new_key]
    return new_weight_keys

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

def numpy_2_torch(numpy_array_file:str,checkpoint_path:str=None):
    numpy_data_dict = np.load(numpy_array_file)
    keys_replacement = absolutely_dbnet_state_dict_keys_replacement()
    db_net = DBNet()
    state_dict = {}
    keys_dict = {}
    for src_key,dst_key in keys_replacement:
        keys_dict[src_key] = dst_key
    weight_keys = numpy_data_dict.files
    # keys_dict = get_keys_dict(weight_keys=weight_keys)
    for key in weight_keys:
        data = numpy_data_dict[key]
        if key not in keys_dict:
            error_info = "can not find key '{}'in paddle weight file!".format(key)
            raise KeyError(error_info)
        new_key = keys_dict[key]
        print("set weight from '{}' to '{}'".format(key,new_key))
        new_key = keys_dict[key]
        state_dict[new_key] = torch.tensor(data,dtype=torch.float32)
    print("successfully load all the keys...")
    db_net.load_state_dict(state_dict)
    db_net.eval()
    image_path = "real_world_text_images/street.png"
    image = cv2.imread(image_path)
    image = image_preprocess(image)
    with torch.no_grad():
        input_tensor = torch.tensor(image, dtype=torch.float32)
        pred = db_net(input_tensor).numpy()
    print(pred)
    torch.save(db_net.state_dict(),checkpoint_path)


if __name__ == "__main__":
    numpy_array_file = "models/db_net_weight.npz"
    checkpoint_path = "models/db_net.pth"
    numpy_2_torch(
        numpy_array_file=numpy_array_file,
        checkpoint_path=checkpoint_path
    )
    
    
        
    
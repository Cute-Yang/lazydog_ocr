import cv2
import numpy as np
import onnxruntime as ort
import paddle
import torch
from numpy.typing import NDArray
from torch import nn

from backbones.mobilenetV3 import MobileNetV3
from experimental import dbnet_by_paddle
from head.db_head import DBHead
from neck.rsefpn import RSEFPN

"""
compare result
backbone: paddle result == torch result
neck: paddle result != torch result
"""

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

def compute_resize_info(h:int,w:int,max_side_len:int=960):
    max_side = max(h,w)
    scale = 1.0
    if max_side > max_side_len:
        scale = max_side_len / max_side
    h = h * scale
    w = w * scale
    # 因为fpn层要求我们的输入至少能够被下采样至 1 / 32,所以要求我们的大小必须是32k
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

def paddel_infernece(state_dict_path:str=None,image:NDArray=None):
    """
    Args:
        state_dict_path:str,the weight path of paddle model
        image:ndarray
    """
    db_net = dbnet_by_paddle.PaddleDBNet()
    state_dict = paddle.load(state_dict_path)
    db_net.set_state_dict(state_dict)
    db_net.eval()
    with paddle.no_grad():
        input_tensor = paddle.to_tensor(image,dtype=paddle.float32)
        pred = db_net(input_tensor)["maps"].numpy()
    return pred

def torch_inference(state_dict_path:str=None,image:NDArray=None):
    """
    Args:
        save as above
    """
    db_net = DBNet()
    state_dict = torch.load(state_dict_path)
    db_net.load_state_dict(state_dict)
    db_net.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(image,dtype=torch.float32)
        pred = db_net(input_tensor).numpy()
    return pred

def onnx_inference(onnx_file:str=None,image:NDArray=None):
    ort_session = ort.InferenceSession(onnx_file)
    feed_dict = {
        "image":image
    }
    output = ort_session.run([],input_feed=feed_dict)
    return output[0]


if __name__ == "__main__":
    image_path = "real_world_text_images/street.png"
    image = cv2.imread(image_path)
    processed_image = image_preprocess(image=image)
    # print(processed_image)
    # 这个是小模型,其实还放出了一个比较大的基于Resnet50的模型
    paddel_state_dict = "models/ch_PP-OCRv3_det_distill_train/student.pdparams"
    paddel_result = paddel_infernece(
        state_dict_path=paddel_state_dict,
        image=processed_image
    )
    checkpoint = "models/db_net.pth"
    torch_result = torch_inference(
        state_dict_path=checkpoint,
        image=processed_image
    )

    onnx_file = "models/mobilenetV3_backbone_set_alpha_0.2_in_hardsigmoid_from_pytorch.onnx"
    ort_result = onnx_inference(onnx_file=onnx_file,image=processed_image)

    print("paddle result:{}".format(paddel_result))
    print("torch result:{}".format(torch_result))
    print("ort output:{}".format(ort_result))
    print(np.sum(paddel_result))
    print(np.sum(torch_result))
    print(np.sum(ort_result))
    s = np.squeeze(ort_result)
    ort_image = cv2.imwrite("ort_image.png",(s * 255.0).astype(np.uint8))
    s = np.squeeze(torch_result)
    ort_image = cv2.imwrite("torch_image.png",(s * 255.0).astype(np.uint8))
    s = np.squeeze(paddel_result)
    ort_image = cv2.imwrite("paddel_image.png",(s * 255.0).astype(np.uint8))
    
    diff = np.sum(np.abs(torch_result - ort_result))
    print("the diff between torch and ort is {}".format(diff))
    diff2 = np.sum(np.abs(paddel_result - ort_result))
    print("the diff between paddle and ort is {}".format(diff2))
    print("export onnx ok...")
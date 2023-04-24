import json

import torch
from torch import nn

from backbones.mobilenetV3 import MobileNetV3
from head.db_head import DBHead
from neck.rsefpn import RSEFPN


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
    # backpone = MobileNetV3(in_channels=3, model_name="large", disable_se=True)
    # print(backpone)
    # backpone.eval()
    # print(backpone.out_channels)
    # 尝试导出一下这个结构看一下
    db_net = DBNet()
    state_dict = torch.load("models/db_net.pth")
    # db_net.load_state_dict(state_dict)
    # torch_model_keys_save_file = "model_keys_compare/db_net_keys_torch.json"
    # with open(torch_model_keys_save_file, "w") as f:
    #     f.write(
    #         json.dumps(
    #             {
    #                 "keys": list(state_dict.keys())
    #             }
    #         )
    #     )
    # save onnx
    save_onnx_flag = True
    save_onnx_file = "db_net.onnx"
    if save_onnx_flag:
        db_net.eval()
        # print(db_net)
        # 根据网络结构可知，这里要求我们的输入的尺寸都是32的倍数
        x = torch.randn(1, 3, 960, 960, dtype=torch.float32)
        _, _, h, w = list(x.shape)
        print("input height:{} input width:{}".format(h, w))
        assert h % 32 == 0, "the height must be divided by 32! for fpn net..."
        assert w % 32 == 0, "the width must be divided by 32! for fpn net"
        with torch.no_grad():
            torch.onnx.export(
                model=db_net,
                args=(x,),
                f=save_onnx_file,
                export_params=True,
                verbose=False,
                input_names=["image"],
                output_names=["shrink_map"],
                dynamic_axes={
                    "image": {
                        0: "batch_size",
                        2: "image_height",
                        3:"image_width"
                    },
                    "shrink_map": {
                        0: "batch_size",
                        2: "image_height",
                        3: "image_width"
                    }
                },
                opset_version=11
            )
            y = db_net(x)
            print(y.numpy())
        try:
            import onnxruntime as ort
            ort_session = ort.InferenceSession(save_onnx_file)
            feed_dict = {
                "image": x.numpy()
            }
            y = ort_session.run([], feed_dict)
            print(y[0])
        except:
            print("failed to verify onnx...")
            import traceback
            traceback.print_exc()


"""
每个stage的shape的变化,假设原来的shape 224
stage 1: 56 1/4
stage 2: 28 1/8
stage 3: 14 1/16
stage 4: 7 1/32
"""

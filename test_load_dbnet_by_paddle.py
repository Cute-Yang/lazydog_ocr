import json

import paddle
from paddle import nn

from experimental import dbnet_by_paddle

# 这里最好仿照其代码风格来搞


class BaseModel(nn.Layer):
    def __init__(self):
        super(BaseModel, self).__init__()
        in_channels = 3
        self.backbone = dbnet_by_paddle.MobileNetV3(
            in_channels=in_channels,
            disable_se=True
        )
        feats_out_channels_list = self.backbone.out_channels
        self.neck = dbnet_by_paddle.RSEFPN(
            in_channels=feats_out_channels_list,
            out_channels=96
        )
        neck_out_channels = self.neck.out_channels
        self.head = dbnet_by_paddle.DBHead(in_channels=neck_out_channels)

    def forward(self, x):
        output = self.backbone(x)
        output = self.neck(output)
        output = self.head(output)
        return output


if __name__ == "__main__":
    state_dict_path = "models/ch_PP-OCRv3_det_distill_train/student.pdparams"
    dbnet = BaseModel()
    # paddle.save(dbnet.state_dict,"gg.pdparams")
    # model_state_dict = dbnet.state_dict()
    model_state_dict = paddle.load(state_dict_path)
    print(list(model_state_dict.keys()))
    state_dict_keys = list(model_state_dict.keys())
    keep_paddel_weight_key_file = "model_keys_compare/db_net_keys_paddel.json"
    with open(keep_paddel_weight_key_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "keys": state_dict_keys
                }
            )
        )
    dbnet.set_state_dict(model_state_dict)
    print("loading the weight dict successfully!")
    # print(dbnet)

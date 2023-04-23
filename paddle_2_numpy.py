import numpy as np
import paddle


def paddel_weight_2_numpy(state_dict_file:str=None,numpy_array_file:str=None):
    state_dict = paddle.load(state_dict_file)
    numpy_array_dict = {}
    for key in state_dict:
        weight_array = state_dict[key].numpy()
        print("key:{} shape:{}".format(key,weight_array.shape))
        numpy_array_dict[key] = weight_array
    np.savez(numpy_array_file,**numpy_array_dict)

if __name__ == "__main__":
    state_dict_path = "models/ch_PP-OCRv3_det_distill_train/student.pdparams"
    numpy_array_file = "models/db_net_weight.npz"
    paddel_weight_2_numpy(
        state_dict_file=state_dict_path,
        numpy_array_file=numpy_array_file
    )
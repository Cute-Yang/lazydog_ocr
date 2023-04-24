"""
在dbnet的fpn中，SE模块下,paddle使用的hardsigmoid层使用了自定义的slope和offset
slope = 0.2f,but torch中不支持自定义的slope,是写死的,所以我们要在导出的graph中
手动的修改这个值,来保证两者的推理是一模一样的哦
SE模块的运算顺序
averagepool -> conv -> relu -> conv -> hardsigmoid!
"""
from typing import List

import onnx


def modify_hardsigmoid_slope_by_name(node_names: List[str], src_onnx_file: str = None, dst_onnx_file: str = None, new_slopes: List[float] = None) -> None:
    """
    modify the hardsigmoid slope by node names
    Args:
        node_names:list of str,the name of hardsigmoid nodes you will modify!
        src_onnx_file:str,the src onnx file,we will read the onnx model from this 
        dst_onnx_file:str,the dst onnx file you will write to!
    """
    # load the onnx model
    onnx_model = onnx.load(src_onnx_file)
    # parse the graph
    # this is the GraphProto
    graph = onnx_model.graph
    expected_node_type = "HardSigmoid"
    previous_expected_node_type = "Conv"
    previous_previous_expected_node_type = "Relu"
    nodes = graph.node
    # node_names = set(node_names)
    if isinstance(new_slopes,float):
        new_slopes = [new_slopes for _ in range(len(modify_node_names))]
    node_name_2_slope = dict(zip(node_names,new_slopes))
    found_node_names = []
    for i in range(len(nodes)):
        # this is a NodeProto
        node = nodes[i]
        op_type = node.op_type
        node_name = node.name
        if node_name in node_name_2_slope:
            print("we find node '{}' need to change!".format(node_name))
            # verify this node
            assert i > 2, "we will check the previous node op type,and previous previos node op type,but get node idx {}".format(i)
            previous_node_type = nodes[i - 1].op_type
            previous_previous_node_type = nodes[i - 2].op_type
            if previous_node_type != previous_expected_node_type or previous_previous_node_type \
                    != previous_previous_expected_node_type or op_type != expected_node_type:
                error_info = "we just want to modify the hardsigmoid at SE module,the expected graph node is Relu -> Conv -> HardSigmoid,but get {} -> {} -> {}".format(
                    previous_previous_node_type,
                    previous_node_type,
                    op_type
                )
                raise ValueError(error_info)
            print("old node:",node)
            new_slope = node_name_2_slope[node_name]
            node_inputs = node.input
            node_outputs = node.output
            new_node = onnx.helper.make_node(
                expected_node_type,
                inputs=node_inputs,
                outputs=node_outputs,
                name=node_name,
                alpha=new_slope
            )
            print("new_node:",new_node)
            graph.node.remove(node)
            graph.node.insert(i,new_node)
            found_node_names.append(node_name)

    if len(node_names) != len(found_node_names):
        error_info = "some specify nodes can not find in graph,the specify nodes is {},but just find {}".format(
            node_names,
            found_node_names
        )
        raise ValueError(error_info)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model,dst_onnx_file)
    print("successfully modify the onnx file!")


if __name__ == "__main__":
    # 每次导出的节点名都在变化
    modify_node_names = [
        # ins 模块的harsigmoid
        "HardSigmoid_117", "HardSigmoid_125", "HardSigmoid_133", "HardSigmoid_141",
        # inp 模块的hardsigmoid
        "HardSigmoid_158", "HardSigmoid_166", "HardSigmoid_174", "HardSigmoid_182"
    ]
    src_onnx_file = "models/mobilenetV3_backbone_from_pytorch.onnx"
    dst_onnx_file = "models/mobilenetV3_backbone_set_alpha_0.2_in_hardsigmoid_from_pytorch.onnx"
    modify_hardsigmoid_slope_by_name(
        node_names=modify_node_names,
        src_onnx_file=src_onnx_file,
        dst_onnx_file=dst_onnx_file,
        new_slopes=0.2
    )

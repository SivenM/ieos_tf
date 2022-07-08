import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib


def get_graph(model):
    """
    Create and return graph_def from model.
    inputs:
        model: keras model
    outputs:
        frozen_model
        graph_def
    """

    tf_model = tf.function(lambda x: model(x))
    tf_model = tf_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_model = convert_variables_to_constants_v2(tf_model)
    graph_def = frozen_model.graph.as_graph_def()
    return frozen_model, graph_def


def get_node_name_list(graph_def)->list:
    """
    Takes and return node names of graph_def
    """
    layers = [op.name for op in graph_def.node]
    return layers


def print_nodes(graph_def):
    layers = [op.name for op in graph_def.node]
    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 60)
    print('\n')

def clear_graph(graph_def):
    """
    Delete NoOP and Identity nodes
    """
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'NoOp':
            del graph_def.node[i]
    for node in graph_def.node:
        for i in reversed(range(len(node.input))):
            if node.input[i][0] == '^':
                del node.input[i]

    node_name_list = get_node_name_list(graph_def)
    print_nodes(graph_def)
    
    graph_def = optimize_for_inference_lib.optimize_for_inference(
                                                                graph_def,
                                                                [node_name_list[0]],
                                                                [node_name_list[-2]],
                                                                tf.float32.as_datatype_enum
                                                                )
    
    print_nodes(graph_def)
    return graph_def


def save_graph(graph_def, save_path):
    with tf.io.gfile.GFile(save_path, 'wb') as f:
        f.write(graph_def.SerializeToString())

def freeze_model(model_path, save_path):
    model = tf.keras.models.load_model(model_path)
    _, graph_def = get_graph(model)
    graph_def = clear_graph(graph_def)
    save_graph(graph_def, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', '-i', type=str, help='input model path')
    parser.add_argument('--output_model', '-o', type=str, help='output model path')
    args = parser.parse_args()
    freeze_model(args.input_model, args.output_model)

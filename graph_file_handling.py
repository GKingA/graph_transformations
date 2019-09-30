import numpy as np
from graph_nets import utils_tf, utils_np
from graph_transformations.helper_functions import is_valid_graph
import json
import os


def process_line(json_dict, keep_features, existence_as_vector):
    """
    Processes one line os the file
    :param json_dict: The dictionary containing the graph data.
    :param keep_features: Whether to keep all features of the graph. It is advised to do so in case of input graphs.
    :param existence_as_vector: Whether to represent existence feature as a vector with the length of 2 in an individual
                                feature vector. It should be False when processing the input graphs.
    :return: Updated json_dict
    """
    if keep_features:
        json_dict['nodes'] = np.array([[i for i in nodes] for nodes in json_dict['nodes']],
                                      dtype=np.str)
        json_dict['edges'] = np.array([[i for i in edges] for edges in json_dict['edges']],
                                      dtype=np.str)
    else:
        if existence_as_vector:
            json_dict['nodes'] = np.array([[1, 0] if nodes[-1] == 0 else [0, 1] for nodes in
                                           json_dict['nodes']], dtype=np.float32)
            json_dict['edges'] = np.array([[1, 0] if edges[-1] == 0 else [0, 1] for edges in
                                           json_dict['edges']], dtype=np.float32)
        else:
            json_dict['nodes'] = np.array([[nodes[-1]] for nodes in json_dict['nodes']], dtype=np.float32)
            json_dict['edges'] = np.array([[edges[-1]] for edges in json_dict['edges']], dtype=np.float32)
    json_dict['globals'] = np.array([float(globals_) for globals_ in json_dict['globals']], dtype=np.float32)
    json_dict['senders'] = np.array(json_dict['senders'], dtype=np.int32)
    json_dict['receivers'] = np.array(json_dict['receivers'], dtype=np.int32)
    return json_dict


def load_graphs(file_path, train_ratio, keep_features, existence_as_vector=True):
    """
    The function extracts the graphs from the given file
    :param file_path: The path to the graph json file
    :param train_ratio: How much of the data should we use for training. The other part is used for testing.
    :param keep_features: Whether to keep all features of the graph. It is advised to do so in case of input graphs.
    :param existence_as_vector: Whether to represent existence feature as a vector with the length of 2 in an individual
                                feature vector. It should be False when processing the input graphs.
    :return: Training and testing GraphTuples
    """
    graph_dicts = []
    with open(file_path) as json_file:
        line = json_file.readline().strip()
        while line != '' and line is not None:
            json_dict = json.loads(line)
            json_dict = process_line(json_dict, keep_features, existence_as_vector)
            is_valid_graph(json_dict)
            graph_dicts.append(json_dict)
            line = json_file.readline().strip()
    train_dicts = graph_dicts[:int(train_ratio * len(graph_dicts))]
    test_dicts = graph_dicts[int(train_ratio * len(graph_dicts)):]
    graphs_tuple_train = utils_tf.data_dicts_to_graphs_tuple(train_dicts)
    graphs_tuple_test = utils_tf.data_dicts_to_graphs_tuple(test_dicts)
    return graphs_tuple_train, graphs_tuple_test


def generate_graph(file_path, batch_size, keep_features, existence_as_vector=True):
    """
    The function extracts batch_size amount of graph from the given file
    :param file_path: The path to the graph json file
    :param batch_size: The number of graph in each batch.
    :param keep_features: Whether to keep all features of the graph. It is advised to do so in case of input graphs.
    :param existence_as_vector: Whether to represent existence feature as a vector with the length of 2 in an individual
                                feature vector. It should be False when processing the input graphs.
    :yield: GraphTuples of the given size
    """
    graph_dicts = []
    with open(file_path) as json_file:
        while True:
            line = json_file.readline().strip()
            while len(graph_dicts) != batch_size:
                if line == "" or line is None:
                    json_file = open(file_path)
                    line = json_file.readline().strip()
                json_dict = json.loads(line)
                json_dict = process_line(json_dict, keep_features, existence_as_vector)
                is_valid_graph(json_dict)
                graph_dicts.append(json_dict)
                line = json_file.readline().strip()
            graphs_tuple = utils_np.data_dicts_to_graphs_tuple(graph_dicts)
            graph_dicts = []
            yield graphs_tuple


def get_first_batch_graph_dict(path, batch_size, keep_features, existence_as_vector=True):
    """
    Gets batch amount of line from the file.
    :param path: The path to the graph json file
    :param batch_size: Size of the batch
    :param keep_features: Whether to keep all features of the graph. It is advised to do so in case of input graphs.
    :param existence_as_vector: Whether to represent existence feature as a vector with the length of 2 in an individual
                                feature vector. It should be False when processing the input graphs.
    :return: The graph in dictionary form
    """
    list_of_dicts = []
    with open(path) as f:
        for i in range(batch_size):
            line = f.readline().strip()
            if line is None or line == "":
                raise IndexError("Batch size is higher, than file size. "
                                 "Use smaller batches or try training on the whole data.")
            json_dict = json.loads(line)
            json_dict = process_line(json_dict, keep_features, existence_as_vector)
            is_valid_graph(json_dict)
            list_of_dicts.append(json_dict)
    return list_of_dicts


def save_predicted_graphs(path, inputs, outputs):
    """
    Saves the predicted graphs to a jsonl file
    :param path: The path where the file shall be saved
    :param inputs: Training input graphs
    :param outputs: Training output graphs
    """
    inputs_dict = utils_np.graphs_tuple_to_data_dicts(inputs)
    outputs_dict = utils_np.graphs_tuple_to_data_dicts(outputs)
    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode) as output:
        for (in_, out_) in zip(inputs_dict, outputs_dict):
            out_dict = {"nodes": [[i[0], int(np.argmax(o))] for (i, o) in zip(in_["nodes"], out_["nodes"])],
                        "edges": [[i[0], int(np.argmax(o))] for (i, o) in zip(in_["edges"], out_["edges"])],
                        "globals": [float(g) for g in in_["globals"]],
                        "senders": in_["senders"].tolist(),
                        "receivers": in_["receivers"].tolist()}
            print(json.dumps(out_dict), file=output)

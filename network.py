from graph_nets import graphs, utils_np, utils_tf
from graph_nets.demos import models
import numpy as np
import tensorflow as tf
from tensorflow._api.v1.losses import mean_pairwise_squared_error as pairwise_mse

import json


def is_valid_graph(json_dict):
    """
    This function determines whether the graph given in a dictionary format is a valid graph
    :param json_dict: Dictionary containing globals, nodes, edges, senders and receivers
    :exception The code throws an exception if the given dictionary is invalid
    """
    if len(json_dict['edges']) != len(json_dict['senders']) or len(json_dict['edges']) != len(
            json_dict['receivers']) or len(json_dict['receivers']) != len(json_dict['senders']):
        raise ValueError("Sizes are not in correspondence {}, {}, {}".format(len(json_dict['edges']),
                                                                             len(json_dict['senders']),
                                                                             len(json_dict['receivers'])))
    for (s, r) in zip(json_dict['senders'], json_dict['receivers']):
        if s >= len(json_dict['nodes']):
            raise ValueError("Nodes do not contain sender {}".format(s))
        if r >= len(json_dict['nodes']):
            raise ValueError("Nodes do not contain receiver {}".format(r))

    standard_edge_len = len(json_dict['edges'][0])
    for e in json_dict['edges']:
        if len(e) != standard_edge_len:
            raise ValueError("Edge feature sizes are not uniform. len({}) != {}".format(e, standard_edge_len))

    standard_node_len = len(json_dict['nodes'][0])
    for n in json_dict['nodes']:
        if len(n) != standard_node_len:
            raise ValueError("Node feature sizes are not uniform. len({}) != {}".format(n, standard_node_len))


def load_graphs(file_path, train_ratio):
    """
    The function extracts the graphs from the given file
    :param file_path: The path to the graph json file
    :param train_ratio: How much of the data should we use for training. The other part is used for testing.
    :return: Training and testing GraphTuples
    """
    graph_dicts = []
    with open(file_path) as json_file:
        line = json_file.readline().strip()
        while line != '' and line is not None:
            json_dict = json.loads(line)
            json_dict['nodes'] = np.array([[float(i) for i in nodes] for nodes in json_dict['nodes']], dtype=np.float32)
            json_dict['edges'] = np.array([[float(i) for i in edges] for edges in json_dict['edges']], dtype=np.float32)
            json_dict['globals'] = np.array([float(globals_) for globals_ in json_dict['globals']], dtype=np.float32)
            json_dict['senders'] = np.array(json_dict['senders'], dtype=np.int32)
            json_dict['receivers'] = np.array(json_dict['receivers'], dtype=np.int32)
            is_valid_graph(json_dict)
            graph_dicts.append(json_dict)
            line = json_file.readline().strip()
    train_dicts = graph_dicts[:int(train_ratio * len(graph_dicts))]
    test_dicts = graph_dicts[int(train_ratio * len(graph_dicts)):]
    graphs_tuple_train = utils_tf.data_dicts_to_graphs_tuple(train_dicts)
    graphs_tuple_test = utils_tf.data_dicts_to_graphs_tuple(test_dicts)
    return graphs_tuple_train, graphs_tuple_test


def print_graphs_tuple(graphs_tuple):
    """
    Helper function that prints the structure of a GraphTuple
    :param graphs_tuple: GraphTuple instance
    """
    print("Shapes of `GraphsTuple`'s fields:")
    print(graphs_tuple.map(lambda x: x if x is None else x.shape, fields=graphs.ALL_FIELDS))
    print("\nData contained in `GraphsTuple`'s fields:")
    print("globals:\n{}".format(graphs_tuple.globals))
    print("nodes:\n{}".format(graphs_tuple.nodes))
    print("edges:\n{}".format(graphs_tuple.edges))
    print("senders:\n{}".format(graphs_tuple.senders))
    print("receivers:\n{}".format(graphs_tuple.receivers))
    print("n_node:\n{}".format(graphs_tuple.n_node))
    print("n_edge:\n{}".format(graphs_tuple.n_edge))


def loss(target, outputs):
    """
    Calculates pairwise root mean squared error on nodes and edges
    :param target: The target graph
    :param outputs: List of output graphs
    :return: It returns the calculated loss
    """
    loss_ = [tf.sqrt(pairwise_mse(target.nodes, output.nodes)) + tf.sqrt(pairwise_mse(target.edges, output.edges)) for output in outputs]
    return loss_


def compute_accuracy(target, output):
    """
    Computes the accuracy ratio on nodes and edges, and also on the whole correctly predicted graph itself
    :param target: The target graphs
    :param output: The output graphs
    :return: The ratio of correctly predicted nodes and edges, and the correctly predicted full graphs
    """
    target_dicts = utils_np.graphs_tuple_to_data_dicts(target)
    output_dicts = utils_np.graphs_tuple_to_data_dicts(output)
    solved = 0
    corrects = []
    for target_dict, output_dict in zip(target_dicts, output_dicts):
        solved_nodes = False
        solved_edges = False
        num_elements = target_dict["nodes"].shape[0]
        num_edges = target_dict["edges"].shape[0]
        correct_nodes = 0
        correct_edges = 0

        for target_nodes, output_nodes in zip(target_dict["nodes"], output_dict["nodes"]):
            if (target_nodes[0] + 0.5 > output_nodes[0] and target_nodes[0] - 0.5 <= output_nodes[0]) and \
                    ((target_nodes[1] == 1. and output_nodes[1] >= 0.5) or (target_nodes[1] == 0. and
                                                                                    output_nodes[1] < 0.5)):
                correct_nodes += 1
        if correct_nodes == num_elements:
            solved_nodes = True

        for target_edges, output_edges in zip(target_dict["edges"], output_dict["edges"]):
            if (target_edges[0] + 0.5 > output_edges[0] and target_edges[0] - 0.5 <= output_edges[0]) and \
                    ((target_edges[1] == 1. and output_edges[1] >= 0.5) or (target_edges[1] == 0. and
                                                                                    output_edges[1] < 0.5)):
                correct_edges += 1
        if correct_edges == num_edges:
            solved_edges = True

        if solved_edges and solved_nodes:
            solved += 1

        corrects.append((correct_nodes, num_elements))
        corrects.append((correct_edges, num_edges))

    corr_, all_ = 0, 0
    for c in corrects:
        corr_ += c[0]
        all_ += c[1]
    return corr_ / all_, solved / len(target)


def train_model(model, inputs_train, targets_train, inputs_test, targets_test):
    """
    The function trains the given model on the training inputs and calculates the accuracy every epoch
    :param model: The model to train
    :param inputs_train: The GraphTuple input used for training
    :param targets_train: The GraphTuple output used as training target
    :param inputs_test: The GraphTuple input used for testing
    :param targets_test: The GraphTuple output used as test target
    """
    output_train = model(inputs_train, num_processing_steps)
    output_test = model(inputs_test, num_processing_steps)

    loss_train = loss(targets_train, output_train)
    loss_train = sum(loss_train) / num_processing_steps

    loss_test = loss(targets_test, output_test)
    loss_test = loss_test[-1]

    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step = optimizer.minimize(loss_train)

    epochs = 10000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(epochs):
            train_values = sess.run({
                "step": step,
                "inputs": inputs_train,
                "targets": targets_train,
                "loss": loss_train,
                "outputs": output_train
            })
            test_values = sess.run({
                "targets": targets_test,
                "loss": loss_test,
                "outputs": output_test,
            })
            correct_train, solved_train = compute_accuracy(train_values["targets"],
                                                           train_values["outputs"][-1])
            correct_test, solved_test = compute_accuracy(test_values["targets"],
                                                         test_values["outputs"][-1])
            print("Train loss: {}\tCorrect train parts: {}\tCorrectly solved train graphs: {}".format(
                train_values["loss"], correct_train, solved_train))
            print("Test loss: {}\tCorrect test parts: {}\tCorrectly solved test graphs: {}".format(
                test_values["loss"], correct_test, solved_test))


if __name__ == '__main__':
    tf.reset_default_graph()

    sentence_graphs_tuple_train, sentence_graphs_tuple_test = load_graphs('./data/sentences.jsonl', 0.8)
    highlighted_graphs_tuple_train, highlighted_graphs_tuple_test = load_graphs('./data/highlight_sentences.jsonl', 0.8)
    num_processing_steps = 10

    encode_process_decode_model = models.EncodeProcessDecode(edge_output_size=2, node_output_size=2,
                                                             global_output_size=1)

    train_model(encode_process_decode_model, sentence_graphs_tuple_train, highlighted_graphs_tuple_train,
                sentence_graphs_tuple_test, highlighted_graphs_tuple_test)


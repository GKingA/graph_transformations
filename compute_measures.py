import numpy as np
from graph_nets import utils_np


def compute_accuracy(target, output):
    """
    Computes the accuracy ratio on nodes and edges, and also on the whole correctly predicted graph itself
    :param target: The target graphs
    :param output: The output graphs
    :return: The ratio of correctly predicted nodes and edges, and the correctly predicted full graphs
    """
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = [xn == yn, xe == ye]
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved


def compute_accuracy_ratio(target, output, is_categorical=False):
    """
    Computes the accuracy ratio on nodes and edges, and also on the whole correctly predicted graph itself
    :param target: The target graphs
    :param output: The output graphs
    :param is_categorical: Whether the output is categorical (feature size = 1) or not
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
            if not is_categorical and (target_nodes[0] + 0.5 > output_nodes[0] >= target_nodes[0] - 0.5) and \
                    ((target_nodes[1] == 1. and output_nodes[1] >= 0.5) or (target_nodes[1] == 0. and
                                                                                    output_nodes[1] < 0.5)):
                correct_nodes += 1
            elif is_categorical and ((target_nodes[0] == 1. and output_nodes[0] >= 0.5) or
                                     (target_nodes[0] == 0. and output_nodes[0] < 0.5)):
                correct_nodes += 1
        if correct_nodes == num_elements:
            solved_nodes = True

        for target_edges, output_edges in zip(target_dict["edges"], output_dict["edges"]):
            if not is_categorical and (target_edges[0] + 0.5 > output_edges[0] >= target_edges[0] - 0.5) and \
                    ((target_edges[1] == 1. and output_edges[1] >= 0.5) or
                     (target_edges[1] == 0. and output_edges[1] < 0.5)):
                correct_edges += 1
            elif is_categorical and ((target_edges[0] == 1. and output_edges[0] >= 0.5) or
                                     (target_edges[0] == 0. and output_edges[0] < 0.5)):
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

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


def compute_accuracy_on_nodes(target, output):
    """
    Computes the accuracy ratio on nodes, and also on the whole correctly predicted graph (its nodes) itself
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
        c = [xn == yn]
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


def compute_one_tp_tn_fp_fn(results, key, value, expected, calculated):
    """
    Calculates the True Positive, True Negative, False Positive and False Negative values for the given label.
    :param results: Dictionary to contain the calculated values. This will be updated.
    :param key: The label we want to update
    :param value: The value of the label
    :param expected: The expected values in the graph
    :param calculated: The calculated values in the graph
    """
    res = {key: {}}
    res[key]["tp"] = len([i for i, j in zip(expected, calculated) if i == value and i == j])
    res[key]["tn"] = len([i for i, j in zip(expected, calculated) if i != value and i == j])
    res[key]["fp"] = len([i for i, j in zip(expected, calculated) if i == value and i != j])
    res[key]["fn"] = len([i for i, j in zip(expected, calculated) if i != value and i != j])
    f_score = compute_precision_recall_f1(res)
    print("{}\t{}\t{}\t{}".format(key, f_score[key]["precision"], f_score[key]["recall"], f_score[key]["f1"]))
    for key2 in results[key]:
        results[key][key2] += res[key][key2]


def compute_tp_tn_fp_fn(target, output, types):
    """
    Calculates the True Positive, True Negative, False Positive and False Negative values in the batch.
    :param target: The expected values in the graph
    :param output: The calculated values in the graph
    :param types: The labels in the graph. (eq. edges0 means the edges with value 0)
    """
    results = {type_: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for type_ in types}
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    for td, od in zip(tdds, odds):
        print("\tprecision\trecall\tf1")
        for type_ in types:
            compute_one_tp_tn_fp_fn(results, type_, int(type_[-1]),
                                    np.argmax(td[type_[:-1]], axis=-1), np.argmax(od[type_[:-1]], axis=-1))
    return results


def add_tp_tn_fp_fn(to_update, batch_result):
    """
    Adds the batch results to the previous results in place.
    :param to_update: The dictionary containing the sum of the previous values. This will be updated.
    :param batch_result: The dictionary containing the results on one batch.
    """
    for key in to_update:
        for key2 in to_update[key]:
            to_update[key][key2] += batch_result[key][key2]


def compute_precision_recall_f1(tp_tn_fp_fn):
    """
    Computes the precision, recall, and f1 score for each label given the dictionary containing the
    True Positive, True Negative, False Positive and False Negative values
    :param tp_tn_fp_fn: dictionary with the True Positive, True Negative, False Positive and False Negative values
           for each type
    :return: dictionary containing the precision, recall, and f1 score for each label
    """
    results = {}
    for key in tp_tn_fp_fn:
        results[key] = {}
        if tp_tn_fp_fn[key]["tp"] + tp_tn_fp_fn[key]["fn"] == 0:
            results[key]["precision"] = 0
        else:
            results[key]["precision"] = tp_tn_fp_fn[key]["tp"] / (tp_tn_fp_fn[key]["tp"] + tp_tn_fp_fn[key]["fn"])
        if tp_tn_fp_fn[key]["tp"] + tp_tn_fp_fn[key]["fp"] == 0:
            results[key]["recall"] = 0
        else:
            results[key]["recall"] = tp_tn_fp_fn[key]["tp"] / (tp_tn_fp_fn[key]["tp"] + tp_tn_fp_fn[key]["fp"])
        if results[key]["precision"] + results[key]["recall"] == 0:
            results[key]["f1"] = 0
        else:
            results[key]["f1"] = 2 * (results[key]["precision"] * results[key]["recall"]) / \
                                 (results[key]["precision"] + results[key]["recall"])
    return results

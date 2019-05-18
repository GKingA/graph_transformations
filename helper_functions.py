from graph_nets import graphs, utils_np
import matplotlib.pyplot as plt
import networkx as nx


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


def visualize_graph(graph_dict):
    """
    Creates a visualization of the given graph
    :param graph_dict: An instance of a graph dictionary
    """
    graph = {"edges": [], "senders": [], "receivers": [], "nodes": [node[:-1] for node in graph_dict["nodes"] if node[-1] == 1.], "globals": [1.0]}
    for edge, sender, receiver in zip(graph_dict["edges"], graph_dict["senders"], graph_dict["receivers"]):
        if edge[-1] == 1. and graph_dict["nodes"][sender][-1] == 1. and graph_dict["nodes"][receiver][-1] == 1.:
            graph["edges"].append(edge[:-1])
            graph["senders"].append(graph["nodes"].index(graph_dict["nodes"][sender]))
            graph["receivers"].append(graph["nodes"].index(graph_dict["nodes"][receiver]))
    graphs_nx = utils_np.graphs_tuple_to_networkxs(utils_np.data_dicts_to_graphs_tuple([graph]))
    nx.draw_networkx(graphs_nx[0])

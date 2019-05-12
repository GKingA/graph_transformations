from graph_nets import graphs


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

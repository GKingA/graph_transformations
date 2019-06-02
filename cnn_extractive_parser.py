import json
import stanfordnlp
import os
import copy


def feature_appender(nodes, edges, senders, receivers, from_node_features, to_node_features, edge, is_summary=False):
    """
    This function is responsible of appending the correct feature vectors to the lists.
    :param nodes: The list of the node feature vectors.
    :param edges: The list of the edge feature vectors.
    :param senders: The list of the sender node ids.
    :param receivers: The list of the receiver node ids.
    :param from_node_features: Feature vector of the sender node.
    :param to_node_features: Feature vector of the receiver node.
    :param edge: Feature vector of the edge.
    :param is_summary: Whether the input features are
    """
    if not is_summary:
        if from_node_features not in nodes:
            nodes.append(from_node_features)
        if to_node_features not in nodes:
            nodes.append(to_node_features)
    else:
        nodes_wo_last_feature = [n[:-1] for n in nodes]
        if from_node_features[-1] == 1.0:
            if from_node_features[:-1] not in nodes_wo_last_feature:
                nodes.append(from_node_features)
            else:
                index = nodes_wo_last_feature.index(from_node_features[:-1])
                nodes[index] = from_node_features
            if to_node_features[:-1] not in nodes_wo_last_feature:
                nodes.append(to_node_features)
            else:
                index = nodes_wo_last_feature.index(to_node_features[:-1])
                nodes[index] = to_node_features
        else:
            if from_node_features[:-1] not in nodes_wo_last_feature:
                nodes.append(from_node_features)
            else:
                index = nodes_wo_last_feature.index(from_node_features[:-1])
                from_node_features = nodes[index]
            if to_node_features[:-1] not in nodes_wo_last_feature:
                nodes.append(to_node_features)
            else:
                index = nodes_wo_last_feature.index(to_node_features[:-1])
                to_node_features = nodes[index]

    senders.append(nodes.index(from_node_features))
    receivers.append(nodes.index(to_node_features))
    edges.append(edge)


def article_graph_builder(uds, best_ids, dependency_dict, dependency_file, vocab_dict, vocab_file, pos_dict, pos_file,
                          wo_index=True):
    """
    This function is used to build a graph dictionary for the article and the summary from the parsed UD.
    :param uds: List of universal dependency graphs in stanfordnlp format
    :param best_ids: The ids of the sentences that have the best score.
    :param dependency_dict: Dictionary containing the dependency types
    :param dependency_file: A jsonl file for saving the dependencies. Used as backup.
    :param vocab_dict: Dictionary containing the lemmatized words
    :param vocab_file: A jsonl file for saving the words. Used as backup.
    :param pos_dict: Dictionary containing the pos tags
    :param pos_file: A jsonl file for saving the pos tags. Used as backup.
    :param wo_index: Whether to consider the index of word when building the graph
    :return: Graph dictionaries of the article and the summary used to construct graph_nets GraphTuple.
    """
    article_nodes = []
    article_edges = []
    article_senders = []
    article_receivers = []
    summary_nodes = []
    summary_edges = []
    summary_senders = []
    summary_receivers = []
    for id, ud in enumerate(uds):
        for s in ud.sentences:
            for dep in s.dependencies:
                from_node = [dep[0].lemma, dep[0].upos]
                to_node = [dep[2].lemma, dep[2].upos]
                edge = dep[1]
                if from_node[0] not in vocab_dict:
                    vocab_dict[from_node[0]] = len(vocab_dict)
                    vocab_file.write(json.dumps({from_node[0]: vocab_dict[from_node[0]]}))
                    vocab_file.write('\n')
                if to_node[0] not in vocab_dict:
                    vocab_dict[to_node[0]] = len(vocab_dict)
                    vocab_file.write(json.dumps({to_node[0]: vocab_dict[to_node[0]]}))
                    vocab_file.write('\n')

                if from_node[1] not in pos_dict:
                    pos_dict[from_node[1]] = len(pos_dict)
                    pos_file.write(json.dumps({from_node[1]: pos_dict[from_node[1]]}))
                    pos_file.write('\n')
                if to_node[1] not in pos_dict:
                    pos_dict[to_node[1]] = len(pos_dict)
                    pos_file.write(json.dumps({to_node[1]: pos_dict[to_node[1]]}))
                    pos_file.write('\n')

                if edge not in dependency_dict:
                    dependency_dict[edge] = len(dependency_dict)
                    dependency_file.write(json.dumps({edge: dependency_dict[edge]}))
                    dependency_file.write('\n')

                if wo_index:
                    from_node_features = [vocab_dict[from_node[0]], pos_dict[from_node[1]]]
                    to_node_features = [vocab_dict[to_node[0]], pos_dict[to_node[1]]]
                else:
                    from_node_features = [vocab_dict[from_node[0]], pos_dict[from_node[1]], int(dep[0].index)]
                    to_node_features = [vocab_dict[to_node[0]], pos_dict[to_node[1]], int(dep[2].index)]

                feature_appender(article_nodes, article_edges, article_senders, article_receivers,
                                 from_node_features, to_node_features, [dependency_dict[edge]])

                feature_value = 1.0 if id in best_ids else 0.0
                summary_from_node_features = from_node_features + [feature_value]
                summary_to_node_features = to_node_features + [feature_value]
                edge_features = [dependency_dict[edge], feature_value]
                feature_appender(summary_nodes, summary_edges, summary_senders, summary_receivers,
                                 summary_from_node_features, summary_to_node_features, edge_features, is_summary=True)

    article_data_dict = {
        "globals": [1.0],
        "nodes": article_nodes,
        "edges": article_edges,
        "senders": article_senders,
        "receivers": article_receivers
    }
    summary_data_dict = {
        "globals": [1.0],
        "nodes": summary_nodes,
        "edges": summary_edges,
        "senders": summary_senders,
        "receivers": summary_receivers
    }
    return article_data_dict, summary_data_dict


def main():
    """
    The main function. It loads the stanfornlp pipeline and processes the lines in the input path with it and uses these
    results to build the graphs.
    """
    # Initialize
    if not os.path.exists("/home/kinga-gemes/stanfordnlp_resources/en_ewt_models"):
        stanfordnlp.download('en')
    nlp = stanfordnlp.Pipeline()

    dep_vocab = {}
    word_vocab = {}
    pos_vocab = {}

    sent_json = open('./data/sentences0.jsonl', 'w')
    high_json = open('./data/highlights0.jsonl', 'w')
    high_s_json = open('./data/highlight_sentences0.jsonl', 'w')
    deps = open('./data/dep_vocab0.jsonl', 'w')
    word = open('./data/word_vocab0.jsonl', 'w')
    pos = open('./data/pos_vocab0.jsonl', 'w')

    # Process and save
    with open('./data/cnn_dm_i4.jsonl') as cnn_dm:
        line = cnn_dm.readline().strip()
        i = 0
        while line is not None and line != '':
            m = json.loads(line)
            sentences = [nlp(sentence) for sentence in m['sentences']]
            article_dict, summary_dict = article_graph_builder(sentences, m["best_ids"], dep_vocab, deps, word_vocab,
                                                               word, pos_vocab, pos)

            sent_json.write(json.dumps(article_dict))
            high_json.write(json.dumps(summary_dict))

            high_json.write('\n')
            sent_json.write('\n')
            high_s_json.write('\n')
            line = cnn_dm.readline().strip()
            i += 1
            print("{} extractive article processed".format(i))

    sent_json.close()
    high_json.close()
    high_s_json.close()
    deps.close()
    word.close()

    with open('./data/dep_vocab0.json', 'w') as dep_json:
        dep_json.write(json.dumps(dep_vocab))

    with open('./data/word_vocab0.json', 'w') as word_json:
        word_json.write(json.dumps(word_vocab))

    with open('./data/pos_vocab0.json', 'w') as word_json:
        word_json.write(json.dumps(pos_vocab))


if __name__ == '__main__':
    main()

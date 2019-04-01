import json
import stanfordnlp
import os
import copy


def highlight_to_sentences(highlight, article):
    """
    This function constructs a graph dictionary from the article's graph labeling the nodes that also appear in the
    highlight's graph.
    :param highlight: The highlight's graph dictionary.
    :param article: The article's graph dictionary.
    :return: Graph dictionary used to construct graph_nets GraphTuple.
    """
    data_dict = copy.deepcopy(article)
    node_feature_size = len(article["nodes"][0])
    for i, (sender, receiver, edge) in enumerate(zip(article["senders"], article["receivers"], article["edges"])):
        for (h_sender, h_receiver, h_edge) in zip(highlight["senders"], highlight["receivers"], highlight["edges"]):
            if edge == h_edge and article['nodes'][sender][0] == highlight['nodes'][h_sender][0] and \
                            article['nodes'][receiver][0] == highlight['nodes'][h_receiver][0]:
                if len(data_dict['edges']) < 2:
                    data_dict['edges'][i].append(1.0)
                if len(data_dict['nodes'][sender]) == node_feature_size:
                    data_dict['nodes'][sender].append(1.0)
                if len(data_dict['nodes'][receiver]) == node_feature_size:
                    data_dict['nodes'][receiver].append(1.0)
        if len(data_dict['edges'][i]) < 2:
            data_dict['edges'][i].append(0.0)
    for i, n in enumerate(data_dict['nodes']):
        if len(n) == node_feature_size:
            data_dict['nodes'][i].append(0.0)
    return data_dict


def graph_builder(ud, dependency_dict, dependency_file, vocab_dict, vocab_file, wo_index=True):
    """
    This function is used to build a graph dictionary from the parsed UD.
    :param ud: Universal dependency graph in stanfordnlp format
    :param dependency_dict: Dictionary containing the dependency types
    :param dependency_file: A jsonl file for saving the dependencies. Used as backup.
    :param vocab_dict: Dictionary containing the lemmatized words
    :param vocab_file: A jsonl file for saving the words. Used as backup.
    :param wo_index: Whether to consider the index of word when building the graph
    :return: Graph dictionary used to construct graph_nets GraphTuple.
    """
    nodes = []
    edges = []
    senders = []
    receivers = []
    for s in ud.sentences:
        for dep in s.dependencies:
            from_node = dep[0].lemma
            to_node = dep[2].lemma
            edge = dep[1]
            if from_node not in vocab_dict:
                vocab_dict[from_node] = len(vocab_dict)
                vocab_file.write(json.dumps({from_node: vocab_dict[from_node]}))
                vocab_file.write('\n')
            if to_node not in vocab_dict:
                vocab_dict[to_node] = len(vocab_dict)
                vocab_file.write(json.dumps({to_node: vocab_dict[to_node]}))
                vocab_file.write('\n')
            if edge not in dependency_dict:
                dependency_dict[edge] = len(dependency_dict)
                dependency_file.write(json.dumps({edge: dependency_dict[edge]}))
                dependency_file.write('\n')
            if wo_index:
                from_node_features = [vocab_dict[from_node]]
                to_node_features = [vocab_dict[to_node]]
            else:
                from_node_features = [vocab_dict[from_node], int(dep[0].index)]
                to_node_features = [vocab_dict[to_node], int(dep[2].index)]
            if from_node_features not in nodes:
                nodes.append(from_node_features)
            if to_node_features not in nodes:
                nodes.append(to_node_features)

            senders.append(nodes.index(from_node_features))
            receivers.append(nodes.index(to_node_features))
            edges.append([dep_vocab[edge]])

    data_dict = {
        "globals": [1.0],
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers
    }
    return data_dict


if __name__ == '__main__':
    # Initialize
    if not os.path.exists("/home/kinga-gemes/stanfordnlp_resources/en_ewt_models"):
        stanfordnlp.download('en')
    nlp = stanfordnlp.Pipeline()

    dep_vocab = {}
    word_vocab = {}

    sent_json = open('sentences.jsonl', 'w')
    high_json = open('highlights.jsonl', 'w')
    high_s_json = open('highlight_sentences.jsonl', 'w')
    deps = open('dep_vocab.jsonl', 'w')
    word = open('word_vocab.jsonl', 'w')

    # Process and save
    with open('cnn-dm_matched.jsonl') as cnn_dm:
        line = cnn_dm.readline().strip()
        i = 0
        # while line is not None and line != '':
        for _ in range(100):
            m = json.loads(line)
            highlights = nlp(m['highlights'])
            sentences = nlp(m['sentences'])
            highlight_dict = graph_builder(highlights, dep_vocab, deps, word_vocab, word)
            sentence_dict = graph_builder(sentences, dep_vocab, deps, word_vocab, word)
            high_sent = highlight_to_sentences(highlight_dict, sentence_dict)

            high_json.write(json.dumps(highlight_dict))
            sent_json.write(json.dumps(sentence_dict))
            high_s_json.write(json.dumps(high_sent))

            high_json.write('\n')
            sent_json.write('\n')
            high_s_json.write('\n')
            line = cnn_dm.readline().strip()
            i += 1
            print(i)

    sent_json.close()
    high_json.close()
    high_s_json.close()
    deps.close()
    word.close()

    with open('dep_vocab.json', 'w') as dep_json:
        dep_json.write(json.dumps(dep_vocab))

    with open('word_vocab.json', 'w') as word_json:
        word_json.write(json.dumps(word_vocab))


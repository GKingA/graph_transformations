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
    edge_feature_size = len(article["edges"][0])
    for i, (sender, receiver, edge) in enumerate(zip(article["senders"], article["receivers"], article["edges"])):
        for (h_sender, h_receiver, h_edge) in zip(highlight["senders"], highlight["receivers"], highlight["edges"]):
            if edge == h_edge and article['nodes'][sender] == highlight['nodes'][h_sender] and \
                                  article['nodes'][receiver] == highlight['nodes'][h_receiver]:
                if len(data_dict['edges'][i]) == edge_feature_size:
                    data_dict['edges'][i].append(1.0)
                if len(data_dict['nodes'][sender]) == node_feature_size:
                    data_dict['nodes'][sender].append(1.0)
                if len(data_dict['nodes'][receiver]) == node_feature_size:
                    data_dict['nodes'][receiver].append(1.0)
        if len(data_dict['edges'][i]) == edge_feature_size:
            data_dict['edges'][i].append(0.0)
    for i, node in enumerate(article["nodes"]):
        for h_node in highlight["nodes"]:
            if node == h_node and len(data_dict['nodes'][i]) == node_feature_size:
                    data_dict['nodes'][i].append(1.0)
    for i, n in enumerate(data_dict['nodes']):
        if len(n) == node_feature_size:
            data_dict['nodes'][i].append(0.0)
    return data_dict


def graph_builder(ud, dependency_dict, dependency_file, vocab_dict, vocab_file, pos_dict, pos_file, wo_index=True):
    """
    This function is used to build a graph dictionary from the parsed UD.
    :param ud: Universal dependency graph in stanfordnlp format
    :param dependency_dict: Dictionary containing the dependency types
    :param dependency_file: A jsonl file for saving the dependencies. Used as backup.
    :param vocab_dict: Dictionary containing the lemmatized words
    :param vocab_file: A jsonl file for saving the words. Used as backup.
    :param pos_dict: Dictionary containing the pos tags
    :param pos_file: A jsonl file for saving the pos tags. Used as backup.
    :param wo_index: Whether to consider the index of word when building the graph
    :return: Graph dictionary used to construct graph_nets GraphTuple.
    """
    nodes = []
    edges = []
    senders = []
    receivers = []
    for s in ud:
        for dep in s:
            from_node = [dep["sender"]["lemma"], dep["sender"]["upos"]]
            to_node = [dep["receiver"]["lemma"], dep["receiver"]["upos"]]
            edge = dep["edge"]
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
                from_node_features = [vocab_dict[from_node[0]], pos_dict[from_node[1]], int(dep["sender"]["index"])]
                to_node_features = [vocab_dict[to_node[0]], pos_dict[to_node[1]], int(dep["receiver"]["index"])]
            if from_node_features not in nodes:
                nodes.append(from_node_features)
            if to_node_features not in nodes:
                nodes.append(to_node_features)

            senders.append(nodes.index(from_node_features))
            receivers.append(nodes.index(to_node_features))
            edges.append([dependency_dict[edge]])

    data_dict = {
        "globals": [1.0],
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers
    }
    return data_dict


def main(processed_file, sentence_jsonl, highlight_jsonl, dependency_jsonl, words_jsonl, pos_jsonl,
         dependency_json, words_json, pos_json):
    """
    The main function. It uses the already processed lines in the input path and uses them to build the graphs.
    :param processed_file: The file containing a json dict in each line with sentences_ud and best_ids keys
    :param sentence_jsonl: The jsonl file to save the article graphs into.
    :param highlight_jsonl: The jsonl file to save the summary graphs into.
    :param dependency_jsonl: The jsonl file to save the dependency dictionary into.
                             It is used as a backup in case the dependency_json could not be saved.
    :param words_jsonl: The jsonl file to save the word dictionary into.
                        It is used as a backup in case the words_json could not be saved.
    :param pos_jsonl: The jsonl file to save the part-of-speech dictionary into.
                      It is used as a backup in case the pos_json could not be saved.
    :param dependency_json: The json file to save the dependency dictionary into.
    :param words_json: The json file to save the word dictionary into.
    :param pos_json: The json file to save the part-of-speech dictionary into.
    """
    # Initialize

    dep_vocab = {}
    word_vocab = {}
    pos_vocab = {}

    sent_json = open(sentence_jsonl, 'w')
    high_s_json = open(highlight_jsonl, 'w')
    deps = open(dependency_jsonl, 'w')
    word = open(words_jsonl, 'w')
    pos = open(pos_jsonl, 'w')

    # Process and save
    with open(processed_file) as cnn_dm:
        line = cnn_dm.readline().strip()
        i = 0
        while line is not None and line != '':
            m = json.loads(line)
            if m['sentences'] == '' or m['highlights'] == '':
                continue
            highlights = m['highlights_ud']
            sentences = m['sentences_ud']
            highlight_dict = graph_builder(highlights, dep_vocab, deps, word_vocab, word, pos_vocab, pos)
            sentence_dict = graph_builder(sentences, dep_vocab, deps, word_vocab, word, pos_vocab, pos)
            high_sent = highlight_to_sentences(highlight_dict, sentence_dict)
            sent_json.write(json.dumps(sentence_dict))
            high_s_json.write(json.dumps(high_sent))

            sent_json.write('\n')
            high_s_json.write('\n')
            line = cnn_dm.readline().strip()
            i += 1
            print("{} article processed".format(i))

    sent_json.close()
    high_s_json.close()
    deps.close()
    word.close()
    pos.close()

    with open(dependency_json, 'w') as dep_json:
        dep_json.write(json.dumps(dep_vocab))

    with open(words_json, 'w') as word_json:
        word_json.write(json.dumps(word_vocab))

    with open(pos_json, 'w') as word_json:
        word_json.write(json.dumps(pos_vocab))


if __name__ == '__main__':
    main('./data/cnn_dm_matched_processed.jsonl', './data/sentences12.jsonl', './data/highlight_sentences12.jsonl',
         './data/dep_vocab12.jsonl', './data/word_vocab12.jsonl', './data/pos_vocab12.jsonl',
         './data/dep_vocab12.json', './data/word_vocab12.json', './data/pos_vocab12.json')

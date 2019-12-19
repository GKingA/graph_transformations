from gensim.parsing.preprocessing import STOPWORDS
from gensim.summarization import summarizer
import json
import os
from rouge import Rouge

from graph_transformations.cnn_extractive_parser import article_graph_builder
#from graph_transformations.network import predict_one_graph
#from graph_transformations.models.model_with_attention import GraphAttention


def gensim_summarize(text):
    """
    Slightly modified version of the gensim.summarization.summarizer.summarize function
    :param text: article to summarize
    :return:
    """
    sentences = summarizer._clean_text_by_sentences(text)
    corpus = summarizer._build_corpus(sentences)

    most_important_docs = summarizer.summarize_corpus(corpus, ratio=1)
    extracted_sentences = summarizer._extract_important_sentences(sentences, corpus, most_important_docs, None)

    return summarizer._format_results(extracted_sentences, True)


def graph_summarize(summary_graph, sentences, sentences_ud, just_connectivity=False):
    """
    Summarize one instance.
    :param summary_graph: The previously predicted summary graph
    :param sentences: The sentences in the article
    :param sentences_ud: The separate ud graphs for each sentence
    :param just_connectivity: Whether to use the node scores for the sentence score calculations. If this is False, the
    node scores are used
    :return: The sentences in the order of relevance
    """
    sentence_scores = {s: 0 for s in sentences}
    for sentence, sent in zip(sentences_ud, sentences):
        for sentence_part in sentence:
            for connection in sentence_part:
                if connection["sender"]["lemma"] is not None and connection["sender"]["lemma"].lower() not in STOPWORDS:
                    sender_score_list = [i[1] for i in summary_graph["nodes"]
                                         if i[0][0] == connection["sender"]["lemma"]
                                         and i[0][1] == connection["sender"]["upos"]]
                    for sender_score in sender_score_list:
                        sentence_scores[sent] += 1 if just_connectivity else sender_score
                if connection["receiver"]["lemma"] is not None and connection["receiver"]["lemma"].lower() not in STOPWORDS:
                    receiver_score_list = [i[1] for i in summary_graph["nodes"]
                                           if i[0][0] == connection["receiver"]["lemma"]
                                           and i[0][1] == connection["receiver"]["upos"]]
                    for receiver_score in receiver_score_list:
                        sentence_scores[sent] += 1 if just_connectivity else receiver_score
    return sorted(sentence_scores.items(), key=lambda kv: kv[1], reverse=True)


def graph_summarize_node_set(summary_graph, sentences, sentences_ud):
    """
    Summarize one instance.
    :param summary_graph: The previously predicted summary graph
    :param sentences: The sentences in the article
    :param sentences_ud: The separate ud graphs for each sentence
    :return: The sentences in the order of relevance
    """
    sentence_scores = {s: 0 for s in sentences}
    for sentence, sent in zip(sentences_ud, sentences):
        for sentence_part in sentence:
            node_set = set(
                [(connection["sender"]["lemma"], connection["sender"]["upos"]) for connection in sentence_part if
                 connection["sender"]["lemma"] is not None and connection["sender"]["lemma"] not in STOPWORDS] + [
                    (connection["receiver"]["lemma"], connection["receiver"]["upos"]) for connection in sentence_part if
                    connection["receiver"]["lemma"] is not None and connection["receiver"]["lemma"] not in STOPWORDS])
            for node in node_set:
                sentence_scores[sent] += [i[1] for i in summary_graph["nodes"] if i[0][0] == node[0] and i[0][1] == node[1]][0]
            if len(node_set) != 0:
                sentence_scores[sent] /= len(node_set)
    return sorted(sentence_scores.items(), key=lambda kv: kv[1], reverse=True)


def find_next_best(best_ids, extracted, sentences, summary):
    max_rouge = 0
    max_id = 0
    for id, sentence in enumerate(sentences):
        if id not in best_ids:
            current_summary = " ".join(extracted + [sentence])
            current_rouge = rouge.get_scores(current_summary, summary)[0]["rouge-1"]["r"]
            if current_rouge > max_rouge:
                max_rouge = current_rouge
                max_id = id
    return sorted(best_ids + [max_id])


def main(jsonl_path, i, last_i, predictions=None, model=None, checkpoint_file=None):
    """
    The main functionality of the code
    :param jsonl_path: The path to the previously preprocessed sentences
    :param i: The assigned index of the first element
    :param last_i: The last index that we dont take into account
    :param predictions: The file containing the predictions for every article from the file from jsonl_path
    :param model: The model used for the graph summary prediction
    :param checkpoint_file: The pretrained checkpoint file
    :return: None
    """
    pred_file = None
    if predictions is not None:
        pred_file = open(predictions)

    for sent_num in range(1, 11):
        dir_name = "./data/safe/sentence_{}".format(sent_num)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    result_complete = [open("./data/safe/sentence_{}/graph_summary_complete_simple.txt".format(sent_num), "a")
                       for sent_num in range(1, 11)]
    result_edges = [open("./data/safe/sentence_{}/graph_summary_edges_simple.txt".format(sent_num), "a")
                    for sent_num in range(1, 11)]
    result_nodes = [open("./data/safe/sentence_{}/graph_summary_nodes_simple.txt".format(sent_num), "a")
                    for sent_num in range(1, 11)]
    extracted = [open("./data/safe/sentence_{}/extracted_summary.txt".format(sent_num), "a")
                 for sent_num in range(1, 11)]
    gensim = [open("./data/safe/sentence_{}/gensim_summary.txt".format(sent_num), "a")
              for sent_num in range(1, 11)]
    with open(jsonl_path) as jsonl_file:
        line = jsonl_file.readline().strip()
        if pred_file is not None:
            pred_line = pred_file.readline().strip()
        while line is not None and line != '':
            if i > last_i:

                json_line = json.loads(line)
                sentences = json_line["sentences"]
                sentences_ud = json_line["sentences_ud"]
                summary = " ".join(json_line["summary"])
                text = " ".join(sentences)

                if pred_file is not None:
                    summary_graph = json.loads(pred_line)
                else:
                    article_graph, _ = article_graph_builder(sentences_ud, json_line["best_ids"])
                    summary_graph = predict_one_graph(model, checkpoint_file, article_graph)

                graph_summary_complete = graph_summarize(summary_graph, sentences, sentences_ud)
                graph_summary_edges = graph_summarize(summary_graph, sentences, sentences_ud, True)
                graph_summary_nodes = graph_summarize_node_set(summary_graph, sentences, sentences_ud)

                gensim_summary = gensim_summarize(text)

                best_ids = json_line["best_ids"]

                best_sentences = [sentences[i] for i in best_ids]

                for sent_num in range(1, 11):
                    best_gensim_summary = " ".join(gensim_summary[:sent_num])
                    best_graph_summary_complete = " ".join([s[0] for s in graph_summary_complete[:sent_num]])
                    best_graph_summary_edges = " ".join([s[0] for s in graph_summary_edges[:sent_num]])
                    best_graph_summary_nodes = " ".join([s[0] for s in graph_summary_nodes[:sent_num]])
                    if sent_num > len(json_line["best_ids"]):
                        if len(summary) == 0:
                            print("empty summary")
                            best_ids = find_next_best(best_ids, best_sentences, sentences, " ".join(best_sentences))
                        else:
                            best_ids = find_next_best(best_ids, best_sentences, sentences, summary)
                        best_sentences = [sentences[i] for i in best_ids]
                    best_extracted_summary = " ".join(best_sentences[:sent_num])
                    print(best_graph_summary_complete, file=result_complete[sent_num-1])
                    print(best_graph_summary_edges, file=result_edges[sent_num-1])
                    print(best_graph_summary_nodes, file=result_nodes[sent_num-1])
                    print(best_gensim_summary, file=gensim[sent_num-1])
                    print(best_extracted_summary, file=extracted[sent_num-1])
            i += 1
            print("{} line processed".format(i))
            line = jsonl_file.readline().strip()
            if pred_file is not None:
                pred_line = pred_file.readline().strip()
    if pred_file is not None:
        pred_file.close()


if __name__ == '__main__':
    rouge = Rouge()
    jsonl_file_path = "./data/safe/cnn_dm_test.jsonl"
    #graph_attention_model = GraphAttention(edge_output_size=2, node_output_size=2, global_output_size=1)
    checkpoint = "./chkpt_attended/model_checkpoint"
    save_file = "./data/safe/graph_summary_one_nodes.txt"
    prediction_file = "./data/tmp/test_predict_attention.jsonl"
    #prediction_file = "./data/tmp/test_predict.jsonl"
    idx = 0
    last_idx = -1
    main(jsonl_file_path, idx, last_idx, predictions = prediction_file)

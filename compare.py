from gensim.parsing.preprocessing import STOPWORDS
from gensim.summarization import summarizer
import json

from graph_transformations.cnn_extractive_parser import article_graph_builder
from graph_transformations.network import predict_one_graph
from graph_transformations.models.model_with_attention import GraphAttention

from rouge.rouge_score import rouge_n, rouge_l_summary_level


def gensim_summarize(text):
    """
    Slightly modified version of the gensim.summarization.summarizer.summarize function
    :param text: article to summarize
    :return: The sentences ordered by relevance
    """
    sentences = summarizer._clean_text_by_sentences(text)
    corpus = summarizer._build_corpus(sentences)
    most_important_docs = summarizer.summarize_corpus(corpus, ratio=1)
    extracted_sentences = summarizer._extract_important_sentences(sentences, corpus, most_important_docs, None)
    return summarizer._format_results(extracted_sentences, True)


def graph_summarize(summary_graph, sentences, sentences_ud):
    """
    Summarize one instance.
    :param summary_graph: The previously predicted summary graph
    :param sentences: The sentences in the article
    :param sentences_ud: The separate ud graphs for each sentence
    :return: The sentences in the order of relevance
    """
    sentence_scores = {s: 0 for s in sentences}
    for sentence, sent in zip(sentences_ud, sentences):
        averager = 0
        for sentence_part in sentence:
            for connection in sentence_part:
                if connection["sender"]["lemma"] is not None and connection["sender"]["lemma"].lower() not in STOPWORDS:
                    sender_score_list = [i[1] for i in summary_graph["nodes"]
                                         if i[0][0] == connection["sender"]["lemma"]
                                         and i[0][1] == connection["sender"]["upos"]]
                    for sender_score in sender_score_list:
                        sentence_scores[sent] += sender_score
                        averager += 1
                if connection["receiver"]["lemma"] is not None and connection["receiver"]["lemma"].lower() not in \
                        STOPWORDS:
                    receiver_score_list = [i[1] for i in summary_graph["nodes"]
                                           if i[0][0] == connection["receiver"]["lemma"]
                                           and i[0][1] == connection["receiver"]["upos"]]
                    for receiver_score in receiver_score_list:
                        sentence_scores[sent] += receiver_score
                        averager += 1
        sentence_scores[sent] /= averager
    return sorted(sentence_scores.items(), key=lambda kv: kv[1], reverse=True)


def main(jsonl_path, save_path, predictions=None, model=None, checkpoint_file=None):
    """
    The main functionality of the code
    :param jsonl_path: The path to the previously preprocessed sentences
    :param save_path: The path to save the rougr scores and summaries
    :param predictions: The file containing the predictions for every article from the file from jsonl_path
    :param model: The model used for the graph summary prediction
    :param checkpoint_file: The pretrained checkpoint file
    :return: None
    """
    pred_file = None
    if predictions is not None:
        pred_file = open(predictions)
    with open(save_path, "a") as result_file:
        with open(jsonl_path) as jsonl_file:
            line = jsonl_file.readline().strip()
            while line is not None and line != '':
                json_line = json.loads(line)
                sentences = json_line["sentences"]
                sentences_ud = json_line["sentences_ud"]
                text = " ".join(sentences)
                gensim_summary = gensim_summarize(text)
                if pred_file is not None:
                    summary_graph = json.loads(pred_file.readline())
                else:
                    article_graph, _ = article_graph_builder(sentences_ud, json_line["best_ids"])
                    summary_graph = predict_one_graph(model, checkpoint_file, article_graph)
                graph_summary = graph_summarize(summary_graph, sentences, sentences_ud)
                number_of_sentences = len(json_line["best_ids"])
                summary = " ".join(json_line["summary"])

                best_gensim_summary = " ".join(gensim_summary[:number_of_sentences])
                best_graph_summary = " ".join([s[0] for s in graph_summary[:number_of_sentences]])
                best_sentences = [sentences[i] for i in json_line["best_ids"]]
                best_extracted_summary = " ".join(best_sentences)
                result = {"gensim": {"rouge-1": rouge_n(best_gensim_summary, summary, n=1),
                                     "rouge-2": rouge_n(best_gensim_summary, summary),
                                     "rouge-l": rouge_l_summary_level(best_gensim_summary, summary),
                                     "summary": best_gensim_summary},
                          "graph": {"rouge-1": rouge_n(best_graph_summary, summary, n=1),
                                    "rouge-2": rouge_n(best_graph_summary, summary),
                                    "rouge-l": rouge_l_summary_level(best_graph_summary, summary),
                                    "summary": best_graph_summary},
                          "extracted": {"rouge-1": rouge_n(best_extracted_summary, summary, n=1),
                                        "rouge-2": rouge_n(best_extracted_summary, summary),
                                        "rouge-l": rouge_l_summary_level(best_extracted_summary, summary),
                                        "summary": best_extracted_summary},
                          "summary": summary
                          }
                print(json.dumps(result), file=result_file)
                line = jsonl_file.readline().strip()
    if pred_file is not None:
        pred_file.close()


if __name__ == '__main__':
    jsonl_file_path = "./data/cnn_dm_i4_processed.jsonl"
    graph_attention_model = GraphAttention(edge_output_size=2, node_output_size=2, global_output_size=1)
    checkpoint = "./chkpt4/model_checkpoint"
    save_file = "./data/rouge.jsonl"
    prediction_file = "./data/tmp/predict_chkpt4.jsonl"
    main(jsonl_file_path, save_file, predictions=prediction_file)

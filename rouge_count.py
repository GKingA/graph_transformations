from pyrouge import Rouge155
import os
import sys


def calculate_rouge_on_data(rouge, model_dir, system_dir, save_rouge):
    """
    Calculates the ROUGE score using the perl script
    :param rouge: the Rouge155 object initialized
    :param model_dir: the directory to the reference summaries
    :param system_dir: the directory of the created summaries
    :param save_rouge: where to save the ROUGE scores
    :return: None
    """
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    output_scores = rouge.convert_and_evaluate()
    rouge_file = open(save_rouge, "w")
    print(output_scores, file=rouge_file)
    rouge_file.close()


if __name__ == '__main__':
    r = Rouge155(os.path.join(os.path.dirname(__file__), 'ROUGE', 'pyrouge', 'tools', 'ROUGE-1.5.5'))

    r.system_filename_pattern = 'summary_(\d+).txt'
    r.model_filename_pattern = 'summary_#ID#.txt'

    i = sys.argv[1]
    print("Sentence length: {}".format(i))

    if not os.path.exists("./data/safe/sentence_{}/gensim_summary_ROUGE.txt".format(i)):
        calculate_rouge_on_data(r,
                                "./data/safe/system_summaries_2",
                                "./data/safe/sentence_{}/gensim_summaries".format(i),
                                "./data/safe/sentence_{}/gensim_summaries_ROUGE.txt".format(i))

    if not os.path.exists("./data/safe/sentence_{}/extracted_summaries_ROUGE.txt".format(i)):
        calculate_rouge_on_data(r,
                                "./data/safe/system_summaries_2",
                                "./data/safe/sentence_{}/extracted_summaries".format(i),
                                "./data/safe/sentence_{}/extracted_summaries_ROUGE.txt".format(i))

    if not os.path.exists("./data/safe/sentence_{}/graph_summaries_edges_ROUGE.txt".format(i)):
        calculate_rouge_on_data(r,
                                "./data/safe/system_summaries_2",
                                "./data/safe/sentence_{}/graph_summaries_edges".format(i),
                                "./data/safe/sentence_{}/graph_summaries_edges_ROUGE.txt".format(i))

    if not os.path.exists("./data/safe/sentence_{}/graph_summaries_complete_simple_ROUGE.txt".format(i)):
        calculate_rouge_on_data(r,
                                "./data/safe/system_summaries_2",
                                "./data/safe/sentence_{}/graph_summaries_complete_simple".format(i),
                                "./data/safe/sentence_{}/graph_summaries_complete_simple_ROUGE.txt".format(i))

    if not os.path.exists("./data/safe/sentence_{}/graph_summaries_nodes_simple_ROUGE.txt".format(i)):
        calculate_rouge_on_data(r,
                                "./data/safe/system_summaries_2",
                                "./data/safe/sentence_{}/graph_summaries_nodes_simple".format(i),
                                "./data/safe/sentence_{}/graph_summaries_nodes_simple_ROUGE.txt".format(i))

    if not os.path.exists("./data/safe/sentence_{}/graph_summaries_complete_attended_ROUGE.txt".format(i)):
        calculate_rouge_on_data(r,
                                "./data/safe/system_summaries_2",
                                "./data/safe/sentence_{}/graph_summaries_complete_attended".format(i),
                                "./data/safe/sentence_{}/graph_summaries_complete_attended_ROUGE.txt".format(i))

    if not os.path.exists("./data/safe/sentence_{}/graph_summares_nodes_attended_ROUGE.txt".format(i)):
        calculate_rouge_on_data(r,
                                "./data/safe/system_summaries_2",
                                "./data/safe/sentence_{}/graph_summares_nodes_attended".format(i),
                                "./data/safe/sentence_{}/graph_summares_nodes_attended_ROUGE.txt".format(i))

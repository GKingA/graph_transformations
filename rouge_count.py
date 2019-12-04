from pyrouge import Rouge155
import os


def calculate_rouge_on_data(rouge, model_dir, system_dir, save_rouge):
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    output_scores = rouge.convert_and_evaluate()
    rouge_file = open(save_rouge, "w")
    print(output_scores, file=rouge_file)
    rouge_file.close()


if __name__ == '__main__':
    r = Rouge155(os.path.join(os.path.dirname(__file__), 'pyrouge', 'tools', 'ROUGE-1.5.5'))

    r.system_filename_pattern = 'summary_(\d+).txt'
    r.model_filename_pattern = 'summary_#ID#.txt'

    cleaned_summaries = "./data/safe/system_summaries_cleaned"

    cleaned_gensim = "./data/safe/gensim_summaries_cleaned"
    gensim_cleaned_rouge = "./data/safe/gensim_cleaned_ROUGE.txt"

    cleaned_extracted = "./data/safe/system_summaries_cleaned"
    extracted_cleaned_rouge = "./data/safe/extracted_cleaned_ROUGE.txt"

    cleaned_graph = "./data/safe/graph_summaries_2_cleaned"
    graph_cleaned_rouge = "./data/safe/graph_cleaned_ROUGE.txt"

    cleaned_graph_just_nodes = "./data/safe/graph_summaries_just_nodes_cleaned"
    graph_just_nodes_cleaned_rouge = "./data/safe/graph_just_nodes_cleaned_ROUGE.txt"

    #calculate_rouge_on_data(r, cleaned_summaries, cleaned_gensim, gensim_cleaned_rouge)
    #calculate_rouge_on_data(r, cleaned_summaries, cleaned_extracted, extracted_cleaned_rouge)
    #calculate_rouge_on_data(r, cleaned_summaries, cleaned_graph, graph_cleaned_rouge)
    #calculate_rouge_on_data(r, cleaned_summaries, cleaned_graph_just_nodes, graph_just_nodes_cleaned_rouge)
    calculate_rouge_on_data(r, "./data/safe/system_summaries_2", "./data/safe/graph_summaries_one_nodes", "./data/safe/graph_summaries_one_nodes_ROUGE.txt")

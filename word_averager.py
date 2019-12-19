import matplotlib.pyplot as plt


def average_word_count(summary_file):
    """
    Calculates the average number of words in each line of the summary file
    :param summary_file: The file to analyse
    :return: the average
    """
    with open(summary_file) as summary:
        line = summary.readline()
        number_of_lines = 0
        number_of_words = 0
        while line is not None and line != '':
            number_of_lines += 1
            number_of_words += len(line.split(" "))
            line = summary.readline()
    return number_of_words / number_of_lines


def make_table(file_path):
    """
    Saves the global variables in a latex table
    :param file_path: the path to save the data
    :return: None
    """
    with open(file_path, 'w') as table_file:
        print("\\begin{table}[!ht]\n\t\centering\n\t\\begin{tabular}{| l | c | c | c | c | c | c | c | c | c | c |}\n"
              "\t\t\hline\n\t\t\\textbf{Model}&\\textbf{1}&\\textbf{2}&\\textbf{3}&\\textbf{4}&\\textbf{5}&\\"
              "textbf{6}&\\textbf{7}&\\textbf{8}&\\textbf{9}&\\textbf{10} \\\\ \hline \hline", file=table_file)
        print("&".join(["\\textbf{Maximum}"]+[str(int(round(e))) for e in extracted]), "\\\\ \hline", file=table_file)
        print("&".join(["\\textbf{Textrank}"]+[str(int(round(e))) for e in gensim]), "\\\\ \hline", file=table_file)
        print("&".join(["\\textbf{G. struct}"]+[str(int(round(e))) for e in graph_structure]),
              "\\\\ \hline", file=table_file)
        print("&".join(["\\textbf{S. graph}"]+[str(int(round(e))) for e in graph_wo_attention]),
              "\\\\ \hline", file=table_file)
        print("&".join(["\\textbf{S. node}"]+[str(int(round(e))) for e in graph_just_nodes_wo_attention]),
              "\\\\ \hline", file=table_file)
        print("&".join(["\\textbf{A. graph}"]+[str(int(round(e))) for e in graph_with_attention]),
              "\\\\ \hline", file=table_file)
        print("&".join(["\\textbf{A. node}"]+[str(int(round(e))) for e in graph_just_nodes_with_attention]),
              "\\\\ \hline", file=table_file)
        print("\t\end{tabular}\n\t\caption{The average word counts by the number of sentences in the summary}\n"
              "\t\label{tab:word_count}\n\end{table}", file=table_file)


if __name__ == "__main__":
    gensim = []
    extracted = []
    graph_structure = []
    graph_wo_attention = []
    graph_just_nodes_wo_attention = []
    graph_with_attention = []
    graph_just_nodes_with_attention = []
    x = [i for i in range(1, 11)]
    for i in range(1, 11):
        print("Sentence length: {}".format(i))
        gensim.append(average_word_count("./data/safe/sentence_{}/gensim_summary.txt".format(i)))
        print("gensim: {}".format(gensim[-1]))

        extracted.append(average_word_count("./data/safe/sentence_{}/extracted_summary.txt".format(i)))
        print("extracted: {}".format(extracted[-1]))

        graph_structure.append(average_word_count("./data/safe/sentence_{}/graph_summary_edges_simple.txt".format(i)))
        print("graph just edges: {}".format(graph_structure[-1]))

        graph_wo_attention.append(
            average_word_count("./data/safe/sentence_{}/graph_summary_complete_simple.txt".format(i)))
        print("graph wo attention: {}".format(graph_wo_attention[-1]))

        graph_just_nodes_wo_attention.append(
            average_word_count("./data/safe/sentence_{}/graph_summary_nodes_simple.txt".format(i)))
        print("graph just nodes wo attention: {}".format(graph_just_nodes_wo_attention[-1]))

        graph_with_attention.append(
            average_word_count("./data/safe/sentence_{}/graph_summary_complete_attended.txt".format(i)))
        print("graph with attention: {}".format(graph_with_attention[-1]))

        graph_just_nodes_with_attention.append(
            average_word_count("./data/safe/sentence_{}/graph_summary_nodes_attended.txt".format(i)))
        print("graph just nodes with attention: {}".format(graph_just_nodes_with_attention[-1]))
    plt.title("Average number of words by sentence count")

    ex_plt = plt.scatter(x, extracted, c="r", alpha=0.5, label="maximum")
    gen_plt = plt.scatter(x, gensim, c="g", alpha=0.5, label="textrank")
    str_plt = plt.scatter(x, graph_structure, c="b", alpha=0.5, label="graph structure")

    simple_plt = plt.scatter(x, graph_wo_attention, c="m", alpha=0.5, label="simple graph")
    simple_node_plt = plt.scatter(x, graph_just_nodes_wo_attention, c="y", alpha=0.5, label= "simple graph just nodes")

    att_plt = plt.scatter(x, graph_with_attention, c="c", alpha=0.5, label="attended graph")
    att_node_plt = plt.scatter(x, graph_just_nodes_with_attention, c="k", alpha=0.5, label="attended graph just nodes")

    legend = plt.legend(loc='upper left')

    plt.savefig("word_counts.png")
    plt.show()
    make_table("word_count_table.tex")

import matplotlib.pyplot as plt


def line_appender(file_path, dictionary):
    """
    Find each relevant ROUGE score in the ROUGE file
    :param file_path: the path of the ROUGE score file
    :param dictionary: the dictionary to load the data into
    :return: the updated dictionary
    """
    try:
        with open(file_path) as ROUGE:
            line = ROUGE.readline()
            while line != "" and line is not None:
                if line.startswith("1 ROUGE-1 Average_R"):
                    dictionary["ROUGE-1"].append(100.0 * float(line.split(" ")[3]))
                if line.startswith("1 ROUGE-2 Average_R"):
                    dictionary["ROUGE-2"].append(100.0 * float(line.split(" ")[3]))
                if line.startswith("1 ROUGE-L Average_R"):
                    dictionary["ROUGE-L"].append(100.0 * float(line.split(" ")[3]))
                if line.startswith("1 ROUGE-SU* Average_R"):
                    dictionary["ROUGE-SU"].append(100.0 * float(line.split(" ")[3]))
                line = ROUGE.readline()
    except:
        print("{} not exists".format(file_path))
    return dictionary


def plot_results(key, png):
    """
    Plots the global dicts' values by given key and saves the image
    :param key: the key for the global dicts
    :param png: the path to save the plot
    :return: None
    """
    plt.title("{} scores by number of sentences in a summary".format(key))
    plt.scatter(x[:len(extracted[key])], extracted[key], c="r", alpha=0.5, label="maximum")
    plt.scatter(x[:len(gensim[key])], gensim[key], c="g", alpha=0.5, label="textrank")
    plt.scatter(x[:len(graph_structure[key])], graph_structure[key], c="b", alpha=0.5,
                label="graph structure")

    plt.scatter(x[:len(graph_complete_simple[key])], graph_complete_simple[key], c="m", alpha=0.5,
                label="simple graph")
    plt.scatter(x[:len(graph_nodes_simple[key])], graph_nodes_simple[key], c="y", alpha=0.5,
                label="simple graph just nodes")

    plt.scatter(x[:len(graph_complete_attended[key])], graph_complete_attended[key], c="c", alpha=0.5,
                label="attended graph")
    plt.scatter(x[:len(graph_nodes_attended[key])], graph_nodes_attended[key], c="k", alpha=0.5,
                label="attended graph just nodes")
    plt.legend(loc='lower right')
    plt.axis(option="equal")
    plt.savefig(png)
    plt.show()
    plt.clf()


def make_table(key, file_path):
    """
    Saves the global variables in a latex table
    :param key: the key for the global dicts
    :param file_path: the path to save the data
    :return: None
    """
    with open(file_path, 'w') as table_file:
        print("\\begin{table}[!ht]\n\t\centering\n\t\\begin{tabular}{| l | c | c | c | c | c | c | c | c | c | c |}\n"
              "\t\t\hline\n\t\t\\textbf{Model}&\\textbf{1}&\\textbf{2}&\\textbf{3}&\\textbf{4}&\\textbf{5}&"
              "\\textbf{6}&\\textbf{7}&\\textbf{8}&\\textbf{9}&\\textbf{10} \\\\ \hline \hline", file=table_file)
        print("&".join(["\\textbf{Maximum}"] + [str(round(e, 2)) for e in extracted[key]]), "\\\\ \hline",
              file=table_file)
        print("&".join(["\\textbf{Textrank}"] + [str(round(e, 2)) for e in gensim[key]]), "\\\\ \hline",
              file=table_file)
        print("&".join(["\\textbf{G. struct}"] + [str(round(e, 2)) for e in graph_structure[key]]), "\\\\ \hline",
              file=table_file)
        print("&".join(["\\textbf{S. graph}"] + [str(round(e, 2)) for e in graph_complete_simple[key]]),
              "\\\\ \hline", file=table_file)
        print("&".join(["\\textbf{S. node}"] + [str(round(e, 2)) for e in graph_nodes_simple[key]]),
              "\\\\ \hline", file=table_file)
        print("&".join(["\\textbf{A. graph}"] + [str(round(e, 2)) for e in graph_complete_attended[key]]),
              "\\\\ \hline", file=table_file)
        print("&".join(["\\textbf{A. node}"] + [str(round(e, 2)) for e in graph_nodes_attended[key]]),
              "\\\\ \hline", file=table_file)
        print("\t\end{{tabular}}\n\t\caption{{The {0} score by the number of sentences in the summary}}\n"
              "\t\label{{tab:{0}}}\n\end{{table}}".format(key), file=table_file)

if __name__ == "__main__":
    gensim = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "ROUGE-SU": []}
    extracted = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "ROUGE-SU": []}
    graph_structure = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "ROUGE-SU": []}
    graph_complete_simple = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "ROUGE-SU": []}
    graph_nodes_simple = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "ROUGE-SU": []}
    graph_complete_attended = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "ROUGE-SU": []}
    graph_nodes_attended = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "ROUGE-SU": []}
    x = [i for i in range(1, 11)]

    for i in range(1, 11):
        gensim = line_appender("./data/safe/sentence_{}/gensim_summary_ROUGE.txt".format(i), gensim)
        extracted = line_appender("./data/safe/sentence_{}/extracted_summaries_ROUGE.txt".format(i), extracted)
        graph_structure = line_appender("./data/safe/sentence_{}/graph_summaries_edges_ROUGE.txt".format(i),
                                        graph_structure)
        graph_complete_simple = line_appender(
            "./data/safe/sentence_{}/graph_summaries_complete_simple_ROUGE.txt".format(i), graph_complete_simple)
        graph_nodes_simple = line_appender("./data/safe/sentence_{}/graph_summaries_nodes_simple_ROUGE.txt".format(i),
                                           graph_nodes_simple)
        graph_complete_attended = line_appender(
            "./data/safe/sentence_{}/graph_summaries_complete_attended_ROUGE.txt".format(i), graph_complete_attended)
        graph_nodes_attended = line_appender("./data/safe/sentence_{}/graph_summares_nodes_attended_ROUGE.txt".format(i),
                                             graph_nodes_attended)

    print("gensim", gensim)
    print("extracted", extracted)
    print("graph_structure", graph_structure)
    print("graph_complete_simple", graph_complete_simple)
    print("graph_nodes_simple", graph_nodes_simple)
    print("graph_complete_attended", graph_complete_attended)
    print("graph_nodes_attended", graph_nodes_attended)
    # ROUGE-1
    plot_results("ROUGE-1", "ROUGE_1.png")
    make_table("ROUGE-1", "ROUGE_1_table.tex")

    # ROUGE-2
    plot_results("ROUGE-2", "ROUGE_2.png")
    make_table("ROUGE-2", "ROUGE_2_table.tex")

    # ROUGE-L
    plot_results("ROUGE-L", "ROUGE_L.png")
    make_table("ROUGE-L", "ROUGE_L_table.tex")

    # ROUGE-SU
    plot_results("ROUGE-SU", "ROUGE_SU.png")
    make_table("ROUGE-SU", "ROUGE_SU_table.tex")

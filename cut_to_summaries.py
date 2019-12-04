import os


def remove_empty(summary_file, model_files, model_output_dirs, summary_output_dir):
    i = 0
    for model_output_dir in model_output_dirs:
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
    if summary_output_dir is not None:
        if not os.path.exists(summary_output_dir):
            os.makedirs(summary_output_dir)
    system_summary = open(summary_file)
    model_summaries = [open(model_file) for model_file in model_files]
    sys_line = system_summary.readline()
    model_lines = [model_summary.readline() for model_summary in model_summaries]
    while model_lines != ["" for _ in model_summaries] and model_lines != [] and "" not in model_summaries:
        if sys_line != "" and sys_line != "\n":
            for (model_line, model_output_dir) in zip(model_lines, model_output_dirs):
                f = open("{}/summary_{}.txt".format(model_output_dir, i), "w")
                f.write(model_line)
                f.close()
            if summary_output_dir is not None:
                f = open("{}/summary_{}.txt".format(summary_output_dir, i), "w")
                f.write(sys_line)
                f.close()
            i += 1
        sys_line = system_summary.readline()
        model_lines = [model_summary.readline() for model_summary in model_summaries]
    system_summary.close()
    for model_summary in model_summaries:
        model_summary.close()
    print(i)


def cut_to_summary(file_name, directory):
    i = 0
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_name) as to_cut:
        line = to_cut.readline()
        while line != "" and line is not None:
            f = open("{}/summary_{}.txt".format(directory, i), "w")
            f.write(line)
            f.close()
            i += 1
            line = to_cut.readline()
    print(file_name, i)


if __name__ == '__main__':
    """cut_to_summary("./data/safe/gensim_summary.txt", "./data/safe/gensim_summaries")
    cut_to_summary("./data/safe/graph_summary.txt", "./data/safe/graph_summaries")
    cut_to_summary("./data/safe/test_summary.txt", "./data/safe/system_summaries")
    cut_to_summary("./data/safe/test_extracted_summary.txt", "./data/safe/extracted_summaries")
    cut_to_summary("./data/safe/graph_summary_2.txt", "./data/safe/graph_summaries_2")
    cut_to_summary("./data/safe/summary_2.txt", "./data/safe/system_summaries_2")
    cut_to_summary("./data/safe/graph_summary_just_nodes.txt", "./data/safe/graph_summaries_just_nodes")"""
    """remove_empty("./data/safe/test_summary.txt",
                 ["./data/safe/test_extracted_summary.txt", "./data/safe/gensim_summary.txt",
                  "./data/safe/graph_summary_2.txt", "./data/safe/graph_summary_just_nodes.txt"],
                 ["./data/safe/extracted_summaries_cleaned", "./data/safe/gensim_summaries_cleaned",
                  "./data/safe/graph_summaries_2_cleaned", "./data/safe/graph_summaries_just_nodes_cleaned"],
                 "./data/safe/system_summaries_cleaned")"""
    cut_to_summary("./data/safe/graph_summary_one_nodes.txt", "./data/safe/graph_summaries_one_nodes")

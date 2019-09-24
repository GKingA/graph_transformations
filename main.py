import os

from graph_transformations.argument_parser import parser


def preprocess(models_dir, processors, extractive, input_file, article_file, summary_file,
               output_train_files, output_test_files, split_ratio):
    """
    This function goes through the entire preprocessing pipeline resulting in files ready to be used as the input of
    the graph neural network.
    :param models_dir: The path to the stanfordnlp directory.
    :param processors: List of parsers to use. Options: tokenize, mwt, pos, lemma, depparse.
    :param extractive: Whether to make extractive summary or not. If true,the input file should contain the best ids.
    :param input_file: This file should contain the articles and the summaries in a jsonl format.
    :param article_file: This file will contain every article graph.
    :param summary_file: This file will contain every summary graph.
    :param output_train_files: The paths to save the training files.
                               The first parameter is the training input, the second is the expected output.
    :param output_test_files: The paths to save the validation files.
                              The first parameter is the validation input, the second is the expected output.
    :param split_ratio: The ratio of data used for training vs validation.
    :return: None
    """
    from graph_transformations.preprocessor import main as stanford_preprocess
    if extractive:
        from graph_transformations.cnn_extractive_parser import main as cnn_process
    else:
        from graph_transformations.cnn_parser import main as cnn_process
    from graph_transformations.train_test_split import train_test_split
    import stanfordnlp

    if not os.path.exists(models_dir):
        stanfordnlp.download('en', resource_dir=models_dir)
    correct_processors = ["tokenize", "mwt", "pos", "lemma", "depparse"]

    incorrect = [i for i in processors if i not in correct_processors]
    if len(incorrect) != 0:
        raise ValueError("The following processor values are incorrect: {}".format(incorrect))

    if not os.path.exists(input_file):
        raise FileNotFoundError("The input file is not found. {} not found".format(input_file))

    pipeline = stanfordnlp.Pipeline(models_dir=models_dir, processors=processors)
    processed_file = "{}_processed.jsonl".format(os.path.splitext(input_file)[0])
    stanford_preprocess(pipeline, input_file, processed_file)

    dependency_file = "dep.jsonl"
    word_file = "words.jsonl"
    pos_file = "pos.jsonl"
    cnn_process(processed_file, article_file, summary_file, dependency_file, word_file, pos_file,
                dependency_file[:-1], word_file[:-1], pos_file[:-1])

    ratio = int(split_ratio) if split_ratio >= 1.0 else int(split_ratio * 100)
    train_test_split(article_file, output_train_files[0], output_test_files[0], ratio)
    train_test_split(summary_file, output_train_files[1], output_test_files[1], ratio)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == "preprocess":
        preprocess(args.models_dir, args.processors, args.extractive, args.input_file,
                   args.article_file, args.summary_file, args.output_train_files,
                   args.output_test_files, args.train_test_split)

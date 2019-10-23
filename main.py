import os

from graph_transformations.argument_parser import parser


def preprocess(models_dir, processors, extractive, cnn_dm_file, article_file, summary_file,
               output_train_files, output_test_files, split_ratio):
    """
    This function goes through the entire preprocessing pipeline resulting in files ready to be used as the input of
    the graph neural network.
    :param models_dir: The path to the stanfordnlp directory.
    :param processors: List of parsers to use. Options: tokenize, mwt, pos, lemma, depparse.
    :param extractive: Whether to make extractive summary or not. If true,the input file should contain the best ids.
    :param cnn_dm_file: This file should contain the articles and the summaries in a jsonl format.
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

    if not os.path.exists(cnn_dm_file):
        raise FileNotFoundError("The input file is not found. {} not found".format(cnn_dm_file))

    pipeline = stanfordnlp.Pipeline(models_dir=models_dir, processors=processors)
    processed_file = "{}_processed.jsonl".format(os.path.splitext(cnn_dm_file)[0])
    stanford_preprocess(pipeline, cnn_dm_file, processed_file)

    dependency_file = "dep.jsonl"
    word_file = "words.jsonl"
    pos_file = "pos.jsonl"
    cnn_process(processed_file, article_file, summary_file, dependency_file, word_file, pos_file,
                dependency_file[:-1], word_file[:-1], pos_file[:-1])

    ratio = int(split_ratio) if split_ratio >= 1.0 else int(split_ratio * 100)
    train_test_split(article_file, output_train_files[0], output_test_files[0], ratio)
    train_test_split(summary_file, output_train_files[1], output_test_files[1], ratio)


def train(model, model_path, save_prediction, valid_files, use_gpu, use_edges, epoch, batch_size,
          training_steps_per_epoch, validation_steps_per_epoch, train_files, print_steps, early_stopping):
    """
    Trains the model given as the parameter using a data generator
    :param model: The model to train
    :param model_path: Where to save the trained model
    :param save_prediction: The path where the predicted output graphs shall be saved
    :param valid_files: The paths of the files used for validation
    :param use_gpu: Which device to use
    :param use_edges: Whether or not to train on the edges as well as the nodes
    :param epoch: The number of epochs
    :param batch_size: The size of a batch
    :param training_steps_per_epoch: The amount of batches to iterate through in a single epoch during the training
    :param validation_steps_per_epoch: The amount of batches to iterate through in a single epoch during the testing
    :param train_files: The paths of the files used for training
    :param print_steps: Whether to print each batch's report
    :param early_stopping: Whether to use Early stopping
    """
    import tensorflow as tf
    from graph_transformations.network import train_generator

    if len(valid_files) != 2 or len(train_files) != 2:
        raise ValueError("Not enough training or validation file specified.\nTraining: {} instead of 2\n"
                         "Validation: {} instead of 2".format(len(train_files), len(valid_files)))

    if model in ["GraphAttention", "ga", "GA"]:
        from graph_transformations.models.model_with_attention import GraphAttention as Model
    else:
        from graph_nets.demos.models import EncodeProcessDecode as Model

    if use_gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu)
        device = "/device:GPU:{}".format(use_gpu)
    else:
        device = "/device:CPU:0"

    tf.reset_default_graph()
    model_to_train = Model(edge_output_size=2, node_output_size=2, global_output_size=1)
    train_generator(model=model_to_train, epochs=epoch, batch_size=batch_size, steps_per_epoch=training_steps_per_epoch,
                    validation_steps_per_epoch=validation_steps_per_epoch, inputs_train_file=train_files[0],
                    outputs_train_file=train_files[1], inputs_test_file=valid_files[0],
                    outputs_test_file=valid_files[1], output_save_path=save_prediction, save_model_path=model_path,
                    use_edges=use_edges, device=device, print_steps=print_steps, early_stopping=early_stopping)


def test(model, model_path, save_prediction, valid_files, use_gpu, use_edges, batch_size, print_steps):
    """
    Tests the model restored from model path with given test data
    :param model: The model to train
    :param model_path: Where to save the trained model
    :param save_prediction: The path where the predicted output graphs shall be saved
    :param valid_files: The paths of the files used for validation
    :param use_gpu: Which device to use
    :param use_edges: Whether or not to train on the edges as well as the nodes
    :param batch_size: The size of a batch
    :param print_steps: Whether to print each batch's report
    """
    import tensorflow as tf
    from graph_transformations.network import test

    if len(valid_files) != 2:
        raise ValueError("Not enough validation file specified. "
                         "Validation: {} instead of 2".format(len(valid_files)))

    if model in ["GraphAttention", "ga", "GA"]:
        from graph_transformations.models.model_with_attention import GraphAttention as Model
    else:
        from graph_nets.demos.models import EncodeProcessDecode as Model

    if use_gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu)
        device = "/device:GPU:{}".format(use_gpu)
    else:
        device = "/device:CPU:0"

    tf.reset_default_graph()
    test(model=model, checkpoint=model_path, input_data=valid_files[0], target_data=valid_files[1],
         batch_size=batch_size, output_save_path=save_prediction, use_edges=use_edges,
         device=device, print_steps=print_steps)


def predict(model, model_path, save_prediction, input_file, use_gpu, batch_size):
    """
    Predicts the output of the model restored from model path with given data
    :param model: The model to train
    :param model_path: Where to save the trained model
    :param save_prediction: The path where the predicted output graphs shall be saved
    :param input_file: The path of the input file used for prediction
    :param use_gpu: Which device to use
    :param batch_size: The size of a batch
    """
    import tensorflow as tf
    from graph_transformations.network import predict

    if model in ["GraphAttention", "ga", "GA"]:
        from graph_transformations.models.model_with_attention import GraphAttention as Model
    else:
        from graph_nets.demos.models import EncodeProcessDecode as Model

    if use_gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(use_gpu)
        device = "/device:GPU:{}".format(use_gpu)
    else:
        device = "/device:CPU:0"

    tf.reset_default_graph()
    predict(model=model, checkpoint=model_path, data=input_file, batch_size=batch_size,
            output_save_path=save_prediction, device=device)


def visualize(file_path, line_number, save_image, all_displayed, color_ones, use_edges):
    """
    Visualize the line_numberth graph in a file
    :param file_path: The file containing one graph dict each line.
    :param line_number: The number which tells us which line to visualize
    :param save_image: The path where the image will be saved
    :param all_displayed: Whether to display every node and edge and not just the ones with 1 value
    :param color_ones: Whether to use color coding to represent node labels
    :param use_edges: Whether to take the connectivity into account while displaying.
    :return: None
    """
    import json
    if all_displayed:
        if color_ones:
            from graph_transformations.helper_functions import visualize_graph_with_colors as visualize_graph
        else:
            from graph_transformations.helper_functions import visualize_original_graph as visualize_graph
    else:
        from graph_transformations.helper_functions import visualize_graph

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise FileNotFoundError("The input file is not found. {} not found".format(file_path))
    if line_number < 0:
        raise ValueError("The given line number ({}) is not valid.".format(line_number))

    with open(file_path) as graph_file:
        i = 0
        line = graph_file.readline()
        while i != line_number:
            try:
                line = graph_file.readline()
            except IOError:
                raise ValueError("The given line number ({}) is not valid.".format(line_number))
            i += 1
        graph_dict = json.loads(line)
        visualize_graph(graph_dict, save_image, use_edges)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == "preprocess":
        preprocess(args.models_dir, args.processors, args.extractive, args.cnn_dm_file,
                   args.article_file, args.summary_file, args.output_train_files,
                   args.output_test_files, args.train_test_split)
    elif args.mode == "train":
        train(args.model, args.model_path, args.save_prediction, args.valid_files, args.use_gpu, args.use_edges,
              args.epoch, args.batch_size, args.training_steps_per_epoch, args.validation_steps_per_epoch,
              args.train_files, args.print_steps, not args.no_early_stopping)
    elif args.mode == "test":
        test(args.model, args.model_path, args.save_prediction, args.valid_files, args.use_gpu, args.use_edges,
             args.batch_size, args.print_steps)
    elif args.mode == "predict":
        predict(args.model, args.model_path, args.save_prediction, args.input_file, args.use_gpu, args.batch_size)
    elif args.mode == "visualize":
        visualize(args.file_path, args.line, args.save_image, args.all_displayed, args.color_ones, args.use_edges)

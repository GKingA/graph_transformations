import argparse
import os

parser = argparse.ArgumentParser(usage='\npython3 main.py preprocess\n'
                                       '[--models_dir MODELS_DIR]\n'
                                       '[--processors PROCESSORS PROCESSORS...]\n'
                                       '[--extractive]\n'
                                       '[--input_file INPUT_FILE]\n'
                                       '[--article_file ARTICLE_FILE]\n'
                                       '[--summary_file SUMMARY_FILE]\n'
                                       '[--output_train_files OUTPUT_TRAIN_FILES OUTPUT_TRAIN_FILES]\n'
                                       '[--output_test_files OUTPUT_TEST_FILES OUTPUT_TEST_FILES]\n'
                                       '[--train_test_split TEST_TRAIN_SPLIT]\n'
                                       '[-v --velocity]\n'
                                       'OR\n'
                                       'python3 main.py train\n'
                                       '[--model {EncodeProcessDecode/epd/EPD, GraphAttention/ga/GA}]\n'
                                       '[--model_path MODEL_PATH]\n'
                                       '[--save_prediction SAVE_PREDICTION, --prediction_file SAVE_PREDICTION]\n'
                                       '[--valid_files VALID_FILES VALID_FILES, '
                                       '--validation_files VALID_FILES VALID_FILES]\n'
                                       '[--use_gpu {-1,0,1}]\n'
                                       '[--use_edges]\n'
                                       '[--epoch EPOCH]\n'
                                       '[--batch_size BATCH_SIZE]\n'
                                       '[--training_steps_per_epoch TRAINING_STEPS_PER_EPOCH, '
                                       '--train_steps_per_epoch TRAINING_STEPS_PER_EPOCH]\n'
                                       '[--validation_steps_per_epoch VALIDATION_STEPS_PER_EPOCH, '
                                       '--valid_steps_per_epoch VALIDATION_STEPS_PER_EPOCH]\n'
                                       '[--train_files TRAIN_FILES TRAIN_FILES]\n'
                                       '[-v --velocity]\n'
                                       '[--no_early_stopping]\n'
                                       'OR\n'
                                       'python3 main.py test\n'
                                       '[--model {EncodeProcessDecode/epd/EPD, GraphAttention/ga/GA}]\n'
                                       '[--model_path MODEL_PATH]\n'
                                       '[--batch_size BATCH_SIZE]\n'
                                       '[--save_prediction SAVE_PREDICTION, --prediction_file SAVE_PREDICTION]\n'
                                       '[--valid_files VALID_FILES VALID_FILES, '
                                       '--validation_files VALID_FILES VALID_FILES]\n'
                                       '[--use_gpu {-1,0,1}]\n'
                                       '[--use_edges]\n'
                                       'OR\n'
                                       'python3 main.py predict\n'
                                       '[--model {EncodeProcessDecode/epd/EPD, GraphAttention/ga/GA}]\n'
                                       '[--model_path MODEL_PATH]\n'
                                       '[--batch_size BATCH_SIZE]\n'
                                       '[--save_prediction SAVE_PREDICTION, --prediction_file SAVE_PREDICTION]\n'
                                       '[--input_file INPUT_FILE]\n'
                                       '[--use_gpu {-1,0,1}]\n'
                                       'OR\n'
                                       'python3 main.py visualize\n'
                                       '--file_path FILE_PATH\n'
                                       '[--line LINE]\n'
                                       '[--save_image SAVE_IMAGE]\n'
                                       '[--use_edges]\n'
                                       '[--all_displayed, --all]'
                                 )
parser.add_argument("mode", choices=["preprocess", "train", "test", "visualize", "predict"])

# Arguments in preprocess mode
DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), "stanfordnlp_resources")
DEFAULT_PROCESSORS = ["tokenize", "mwt", "pos", "lemma", "depparse"]
parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR,
                    help="If you have stanfordnlp_resources write here the path to the directory.")
parser.add_argument("--processors", nargs='*', default=DEFAULT_PROCESSORS,
                    help="Which parsers to use. Options: tokenize, mwt, pos, lemma, depparse. "
                         "Please write them with space as the delimiter")
parser.add_argument("--extractive", action="store_true", default=False,
                    help="Used in preprocess mode. "
                         "If set the summary will only contain exact sentences from the original text.")
parser.add_argument("--input_file", default="./data/cnn_dm_i4.jsonl", type=str,
                    help="Used in preprocess mode. "
                         "The file should contain the articles and the summaries in a jsonl format.")
parser.add_argument("--article_file", default="./data/article.jsonl", type=str,
                    help="Used in preprocess mode. "
                         "This file will contain every article graph.")
parser.add_argument("--summary_file", default="./data/summary.jsonl", type=str,
                    help="Used in preprocess mode. "
                         "This file will contain every summary graph.")
parser.add_argument("--output_train_files", nargs=2,
                    default=["./data/sentences_train2.jsonl", "./data/highlight_sentences_train2.jsonl"],
                    help="Used in preprocess mode. The paths to save the training files.\n"
                         "The first parameter is the training input, the second is the expected output.")
parser.add_argument("--output_test_files", nargs=2,
                    default=["./data/sentences_test2.jsonl", "./data/highlight_sentences_test2.jsonl"],
                    help="Used in preprocess mode. The paths to save the validation files.\n"
                         "The first parameter is the validation input, the second is the expected output.")
parser.add_argument("--train_test_split", default=0.8, type=float,
                    help="Used in preprocess mode. "
                         "This parameter sets the split ratio of the training and the validation set.\n")

# Arguments in train, test and predict mode
parser.add_argument("--model", default="GraphAttention",
                    choices=["EncodeProcessDecode", "epd", "EPD", "GraphAttention", "ga", "GA"],
                    help="Used in train, test and predict mode. This parameter sets the model structure for training.")
parser.add_argument("--model_path", default="model_checkpoint", type=str,
                    help="Used in train, test and predict. In train mode this is the path to save the model.\n"
                         "In test mode this path is used to load the model")
parser.add_argument("--save_prediction", "--prediction_file", default="./data/predictions.jsonl", type=str,
                    help="Used in train, test and predict mode. This file is used to store the predictions.")
parser.add_argument("--use_gpu", default=0, type=int, choices=[-1, 0, 1],
                    help="Used in train, test and predict mode. Sets which GPU to use. "
                         "If there is no GPU available, please set it to -1.")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Used in train, test and predict mode. This parameter determines the size of the batch.")

# Arguments in train and test mode
parser.add_argument("--valid_files", "--validation_files", nargs=2,
                    default=["./data/sentences_test2.jsonl", "./data/highlight_sentences_test2.jsonl"],
                    help="Used in train and test mode. This are the paths to the validation files.\n"
                         " The first argument is the input file, the second is the expected output file.")
parser.add_argument("-v", "--velocity", action='store_true', default=False,
                    help="Used in train and test mode. If set, the F score and accuracy will be displayed "
                         "for each batch and graph.")

# Arguments in predict mode
parser.add_argument("--input_file", default="./data/sentences_test2.jsonl", type=str,
                    help="Used in predict mode. This parameter specifies the path to input file containing graphs,"
                         "which you want to predict the output to.")

# Arguments in train and visualize mode
parser.add_argument("--use_edges", action='store_true', default=False,
                    help="Used in train and visualize mode. This parameter sets the edge usage to true.")

# Arguments in train mode
parser.add_argument("--epoch", default=10, type=int,
                    help="Used in train mode. This parameter determines the number of epochs.")
parser.add_argument("--training_steps_per_epoch", "--train_steps_per_epoch", default=100, type=int,
                    help="Used in train mode. If not set, the default is to use the whole training set in each epoch.")
parser.add_argument("--validation_steps_per_epoch", "--valid_steps_per_epoch", default=100, type=int,
                    help="Used in train mode. If not set, the default is to use the whole validation set in each epoch.")
parser.add_argument("--train_files", nargs=2,
                    default=["./data/sentences_train2.jsonl", "./data/highlight_sentences_train2.jsonl"],
                    help="Used in train mode. This are the paths to the training files.\n"
                         "The first argument is the input file, the second is the expected output file.")
parser.add_argument("--no_early_stopping", default=False, action="store_true",
                    help="Used in train mode. If set, the training won't utilise early stopping.")

# Arguments in visualize mode
parser.add_argument("--file_path", default="", type=str,
                    help="This is a required field in visualize mode.\n"
                         "This determines which file contains the graph you want to visualize.")
parser.add_argument("--line", default=0, type=int,
                    help="Used in visualize mode. You can set which line of the file do you want to visualize as graph")
parser.add_argument("--save_image", default="graph.png", type=str,
                    help="Used in visualize mode. The path to save the image of the graph.")
parser.add_argument("--all_displayed", "--all", action="store_true", default=False,
                    help="Used in visualize mode. If you use this config, the visualization will contain\n"
                         "all nodes and edges even if their value is 0.")

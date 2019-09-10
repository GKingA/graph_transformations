from graph_transformations import preprocessor

import threading
import os
import stanfordnlp
import argparse


if __name__ == "__main__":
    DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), "stanfordnlp_resources")
    DEFAULT_PROCESSORS = "tokenize,mwt,pos,lemma,depparse"

    parser = argparse.ArgumentParser(usage='python3 %(prog) [--models_dir <path> '
                                           '--processors <tokenize,mwt,pos,lemma,depparse>]')
    parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR,
                        help="If you have stanfordnlp_resources write here the path to the directory.")
    parser.add_argument("--processors", type=str, default=DEFAULT_PROCESSORS,
                        help="Which parsers to use. Options: tokenize, mwt, pos, lemma, depparse. "
                             "Please write them with comma as the delimiter")
    args = parser.parse_args()

    if not os.path.exists(args.models_dir):
        stanfordnlp.download('en', resource_dir=args.models_dir)
    pipeline = stanfordnlp.Pipeline(models_dir=args.models_dir, processors=args.processors)
    extractive_thread = threading.Thread(target=preprocessor.main,
                                         args=(pipeline, "./data/cnn_dm_i4.jsonl", "./data/cnn_dm_i4_processed.jsonl"))
    cnn_thread = threading.Thread(target=preprocessor.main,
                                  args=(pipeline, "./data/cnn-dm_matched.jsonl", "./data/cnn_dm_matched_processed.jsonl"))
    extractive_thread.start()
    cnn_thread.start()
    extractive_thread.join()
    cnn_thread.join()

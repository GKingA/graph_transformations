from graph_transformations import preprocessor

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
    parser.add_argument("--from_line", type=int, default=0, help="From which line to start the preprocessing")
    parser.add_argument("--to_line", type=int, default=None, help="To which line to preprocess")
    args = parser.parse_args()

    if not os.path.exists(args.models_dir):
        stanfordnlp.download('en', resource_dir=args.models_dir)
    pipeline = stanfordnlp.Pipeline(models_dir=args.models_dir, processors=args.processors)
    preprocessor.main(pipeline, "./data/cnn_dm_i4.jsonl",
                      "./data/cnn_dm_i4_processed_part{}_{}.jsonl".format(args.from_line, args.to_line),
                      from_line=args.from_line, to_line=args.to_line)

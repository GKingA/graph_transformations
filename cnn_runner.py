from graph_transformations import cnn_parser, cnn_extractive_parser

import threading

if __name__ == "__main__":
    extractive_thread = threading.Thread(target=cnn_extractive_parser.main)
    cnn_thread = threading.Thread(target=cnn_parser.main)
    extractive_thread.start()
    cnn_thread.start()
    extractive_thread.join()
    cnn_thread.join()

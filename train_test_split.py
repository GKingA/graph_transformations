def train_test_split(file_path, train_path, test_path, ratio_in_hundred):
    """
    Splits the input file into training ant test sets
    :param file_path: The input file
    :param train_path: The path to save the training set
    :param test_path: The path to save the test set
    :param ratio_in_hundred: The amount of data from 100 line used for training
    """
    with open(file_path) as input_file:
        train = open(train_path, 'w')
        test = open(test_path, 'w')
        line = input_file.readline().strip()
        while line != '' and line is not None:
            for _ in range(ratio_in_hundred):
                print(line, file=train)
                line = input_file.readline().strip()
                if line == '' or line is None:
                    break
            for _ in range(100 - ratio_in_hundred):
                print(line, file=test)
                line = input_file.readline().strip()
                if line == '' or line is None:
                    break
        train.close()
        test.close()


if __name__ == "__main__":
    train_test_split("./data/sentences.jsonl", "./data/sentences_train.jsonl", "./data/sentences_test.jsonl", 80)
    train_test_split("./data/highlight_sentences.jsonl", "./data/highlight_sentences_train.jsonl",
                     "./data/highlight_sentences_test.jsonl", 80)

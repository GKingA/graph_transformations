import json


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


def update(input_path, length):
    print("Correction")
    data = []
    with open(input_path) as input_file:
        for i in range(length):
            data.append(input_file.readline())
    with open(input_path, "w") as input_file:
        print("".join(data), file=input_file)


def test_split(input_path, output_path):
    print(input_path)
    with open(input_path) as input_file:
        with open(output_path) as output_file:
            input_ = input_file.readline()
            output_ = output_file.readline()
            i = 0
            while input_ != '' and input_ is not None and output_ != '' and output_ is not None:
                inp = json.loads(input_)
                out = json.loads(output_)
                if len(inp["edges"]) == len(out["edges"]) and len(inp["nodes"]) == len(out["nodes"]):
                    print(i)
                else:
                    print("ERROR on line {}".format(i + 1))
                    print(len(inp["edges"]), len(out["edges"]), len(inp["nodes"]), len(out["nodes"]))
                input_ = input_file.readline()
                output_ = output_file.readline()
                i += 1
            try:
                input_file.readline()
                update(input_path, i)
            except:
                print("File lengths are good")


if __name__ == "__main__":
    train_test_split("./data/sentences.jsonl", "./data/sentences_train.jsonl", "./data/sentences_test.jsonl", 80)
    train_test_split("./data/highlight_sentences.jsonl", "./data/highlight_sentences_train.jsonl",
                     "./data/highlight_sentences_test.jsonl", 80)
    test_split("./data/sentences_train.jsonl", "./data/highlight_sentences_train.jsonl")
    test_split("./data/sentences_test.jsonl", "./data/highlight_sentences_test.jsonl")

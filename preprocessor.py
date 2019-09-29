import json


def dependency_parse(nlp, article):
    """
    Processes a line of data
    :param nlp: The stanfordnlp pipeline to process the data
    :param article: An article which can be in a string or a list format
    :return: Parsed article
    """
    if type(article) == str:
        parsed = nlp(article)
        return [[{"sender": {k[1:]: dep[0].__dict__[k] for k in dep[0].__dict__ if not k.startswith("_parent")},
                  "edge": dep[1],
                  "receiver": {k[1:]: dep[2].__dict__[k] for k in dep[2].__dict__ if not k.startswith("_parent")}}
                 for dep in sentence.dependencies] for sentence in parsed.sentences]
    elif type(article) == list:
        parsed = [nlp(sentence) for sentence in article]
        return [[[{"sender": {k[1:]: dep[0].__dict__[k] for k in dep[0].__dict__ if not k.startswith("_parent")},
                   "edge": dep[1],
                   "receiver": {k[1:]: dep[2].__dict__[k] for k in dep[2].__dict__ if not k.startswith("_parent")}}
                  for dep in sentence.dependencies] for sentence in p.sentences] for p in parsed]


def process_line(nlp, line, parsed_file):
    """
    The step repeated in the main function. It processes a single line and writes the result to the final file
    :param nlp: The stanfordnlp pipeline to process the data
    :param line: The current line to process
    :param parsed_file: The final file containing the results
    """
    m = json.loads(line)
    if "highlights" in m:
        if m['sentences'] != '' and m['highlights'] != '':
            m["highlights_ud"] = dependency_parse(nlp, m['highlights'])
            m["sentences_ud"] = dependency_parse(nlp, m['sentences'])
    else:
        if m['sentences'] != '':
            m["sentences_ud"] = dependency_parse(nlp, m['sentences'])
    parsed_file.write(json.dumps(m))


def main(nlp, file_path, final_file_path, from_line=0, to_line=None):
    """
    Main function to process the input file with the nlp pipeline.
    :param nlp: The stanfordnlp pipeline to process the data
    :param file_path: The file that contains the data to process
    :param final_file_path: The file to save the processed data
    :param from_line: The first line to process
    :param to_line: The first line that shall not be processed. If None, we shall process it until the end of the file
    """
    with open(final_file_path, "w") as parsed_file:
        with open(file_path) as cnn_dm:
            line = cnn_dm.readline().strip()
            article_idx = 0
            while article_idx < from_line:
                line = cnn_dm.readline().strip()
                article_idx += 1
            if to_line is None:
                while line is not None and line != '':
                    process_line(nlp, line, parsed_file)
                    article_idx += 1
                    print("{} articles processed from file {}".format(article_idx, file_path))
                    line = cnn_dm.readline().strip()
            else:
                while article_idx < to_line and line is not None and line != '':
                    process_line(nlp, line, parsed_file)
                    article_idx += 1
                    print("{}th article processed from file {}".format(article_idx, file_path))
                    line = cnn_dm.readline().strip()

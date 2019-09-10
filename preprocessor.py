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


def main(nlp, file_path, final_file_path):
    """
    Main function to process the input file with the nlp pipeline.
    :param nlp: The stanfordnlp pipeline to process the data
    :param file_path: The file that contains the data to process
    :param final_file_path: The file to save the processed data
    """
    with open(final_file_path, "w") as parsed_file:
        with open(file_path) as cnn_dm:
            line = cnn_dm.readline().strip()
            article_idx = 0
            while line is not None and line != '':
                m = json.loads(line)
                if "highlights" in m:
                    if m['sentences'] == '' or m['highlights'] == '':
                        continue
                    m["highlights_ud"] = dependency_parse(nlp, m['highlights'])
                    m["sentences_ud"] = dependency_parse(nlp, m['sentences'])
                else:
                    if m['sentences'] == '':
                        continue
                    m["sentences_ud"] = dependency_parse(nlp, m['sentences'])
                parsed_file.write(json.dumps(m))
                article_idx += 1
                print("{} articles processed from file {}".format(article_idx, file_path))

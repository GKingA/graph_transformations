from graph_nets import utils_tf, utils_np
from graph_nets.demos.models import EncodeProcessDecode
import tensorflow as tf
import os

from graph_transformations.models.model_with_attention import GraphAttention
from graph_transformations.graph_losses import softmax_loss, softmax_loss_on_nodes
from graph_transformations.graph_file_handling import get_first_batch_graph_dict, generate_graph, \
                                                      save_predicted_graphs, process_line
from graph_transformations.compute_measures import compute_accuracy, compute_accuracy_on_nodes, \
                                                   compute_tp_tn_fp_fn, add_tp_tn_fp_fn, compute_precision_recall_f1


def generate_placeholder(file, batch_size, keep_features):
    """
    Generates a placeholder
    :param file: The path to the graph json file
    :param batch_size: Size of the batch
    :param keep_features: Whether to keep all features of the graph. It is advised to do so in case of input graphs.
    :return: Generated placeholder
    """
    return utils_tf.placeholders_from_data_dicts(get_first_batch_graph_dict(file, batch_size, keep_features))


def train_model(model, epochs, inputs_train, targets_train, inputs_test, targets_test, output_save_path):
    """
    The function trains the given model on the training inputs and calculates the accuracy every epoch
    :param model: The model to train
    :param epochs: The number of training iterations
    :param inputs_train: The GraphTuple input used for training
    :param targets_train: The GraphTuple output used as training target
    :param inputs_test: The GraphTuple input used for testing
    :param targets_test: The GraphTuple output used as test target
    :param output_save_path: The path where the final output graphs shall be saved
    """
    output_train = model(inputs_train, num_processing_steps)
    output_test = model(inputs_test, num_processing_steps)

    loss_train = softmax_loss(targets_train, output_train)
    loss_train = sum(loss_train) / num_processing_steps

    loss_test = softmax_loss(targets_test, output_test)
    loss_test = loss_test[-1]

    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step = optimizer.minimize(loss_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(epochs):
            train_values = sess.run({
                "step": step,
                "inputs": inputs_train,
                "targets": targets_train,
                "loss": loss_train,
                "outputs": output_train
            })
            test_values = sess.run({
                "inputs": inputs_test,
                "targets": targets_test,
                "loss": loss_test,
                "outputs": output_test,
            })
            correct_train, solved_train = compute_accuracy(train_values["targets"], train_values["outputs"][-1])
            correct_test, solved_test = compute_accuracy(test_values["targets"], test_values["outputs"][-1])
            print("{}th epoch".format(iteration))
            print("Train loss: {}\tCorrect train parts: {}\tCorrectly solved train graphs: {}".format(
                train_values["loss"], correct_train, solved_train))
            print("Test loss: {}\tCorrect test parts: {}\tCorrectly solved test graphs: {}".format(
                test_values["loss"], correct_test, solved_test))
        save_predicted_graphs(output_save_path, train_values["inputs"], train_values["outputs"][-1])
        save_predicted_graphs(output_save_path, test_values["inputs"], test_values["outputs"][-1])


def run_session(sess, inputs, input_ph, outputs, steps_per_epoch,
                use_edges=None, targets=None, target_ph=None, optimize=None, loss=None, print_steps=None,
                output_save_path=None, accurately=False):
    """
    Runs training, testing or prediction in session
    :param sess: The tensorflow session for running the training testing or prediction.
    :param inputs: The generator of the inputs
    :param input_ph: Placeholders for the inputs
    :param outputs: The outputs of the network
    :param steps_per_epoch: The amount of batches to iterate through in a single epoch
    :param use_edges: Whether or not to train/test on the edges as well as the nodes
    :param targets: The generator of the target outputs
    :param target_ph: Placeholders for the targets
    :param optimize: Function for optimizing the network
    :param loss: Loss function
    :param print_steps: Whether to print each batch's report
    :param output_save_path: The path where the final output graphs shall be saved
    :param accurately: Whether to save the scores or just whether the result was one or zero
    :return: The function returns the current loss, if it could be calculated
    """
    losses = []
    corrects = []
    solved = []

    types = ["edges0", "edges1", "nodes0", "nodes1"] if use_edges else ["nodes0", "nodes1"]
    tp_tn_fp_fn = {type_: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for type_ in types}

    step = 0
    sess_dict = {"inputs": input_ph, "outputs": outputs}
    mode = "Predict"
    iterator = inputs
    if optimize is not None:
        mode = "Train"
        sess_dict = {
            "optimize": optimize,
            "inputs": input_ph,
            "targets": target_ph,
            "loss": loss,
            "outputs": outputs
        }
        iterator = zip(inputs, targets)
    elif targets is not None:
        mode = "Test"
        sess_dict = {
            "inputs": input_ph,
            "targets": target_ph,
            "loss": loss,
            "outputs": outputs
        }
        iterator = zip(inputs, targets)

    for batch in iterator:
        if len(batch) == 2:
            feed_dict = {input_ph: batch[0], target_ph: batch[1]}
        else:
            feed_dict = {input_ph: batch}

        values = sess.run(sess_dict, feed_dict=feed_dict)

        if output_save_path is not None:
            save_predicted_graphs(output_save_path, values["inputs"], values["outputs"][-1], accurately)

        if targets is not None:
            correct, solved_ = compute_accuracy(values["targets"], values["outputs"][-1]) if use_edges \
                else compute_accuracy_on_nodes(values["targets"], values["outputs"][-1])

            if print_steps:
                print("{mode} loss: {0}\tCorrect {mode} parts: {1}\tCorrectly solved {mode} graphs: {2}".format(
                      values["loss"], correct, solved, mode=mode))

            add_tp_tn_fp_fn(tp_tn_fp_fn, compute_tp_tn_fp_fn(values["targets"], values["outputs"][-1], types,
                                                             print_steps))

            losses.append(values["loss"])
            corrects.append(correct)
            solved.append(solved_)
        if step < steps_per_epoch:
            step += 1
        else:
            break

    if targets is not None:
        current_loss = sum(losses) / steps_per_epoch
        print("{} loss: {}\tCorrect test parts: {}\tCorrectly solved test graphs: {}".format(mode,
              current_loss, sum(corrects) / steps_per_epoch, sum(solved) / steps_per_epoch))

        report = compute_precision_recall_f1(tp_tn_fp_fn)
        print("{} report\n\t\tprecision\trecall\tf1".format(mode))
        for key in report:
            print("{}\t{}\t{}\t{}".format(key, report[key]["precision"], report[key]["recall"],
                                          report[key]["f1"]))
        return current_loss
    return None


def train_generator(model, epochs, batch_size, steps_per_epoch, validation_steps_per_epoch,
                    inputs_train_file, outputs_train_file,
                    inputs_test_file, outputs_test_file, output_save_path, save_model_path, accurately,
                    use_edges=True, device='/device:GPU:0', print_steps=False, early_stopping=True):
    """
    Trains the model given as the parameter using a data generator
    :param model: The model to train
    :param epochs: The number of training iterations
    :param batch_size: The size of a training batch
    :param steps_per_epoch: The amount of batches to iterate through in a single epoch during the training
    :param validation_steps_per_epoch: The amount of batches to iterate through in a single epoch during the testing
    :param inputs_train_file: The path of the input file used for training
    :param outputs_train_file: The path of the target file used for training
    :param inputs_test_file: The path of the input file used for testing
    :param outputs_test_file: The path of the target file used for testing
    :param output_save_path: The path where the final output graphs shall be saved
    :param save_model_path: Where to save the trained model
    :param accurately: Whether to save the scores or just whether the result was one or zero
    :param use_edges: Whether or not to train on the edges as well as the nodes
    :param device: Which device to use. GPU:0 is the default.
    :param print_steps: Whether to print each batch's report
    :param early_stopping: Whether to use Early stopping
    :return: The trained model
    """

    inputs_train = generate_graph(inputs_train_file, batch_size, keep_features=True)
    inputs_test = generate_graph(inputs_test_file, batch_size, keep_features=True)
    targets_train = generate_graph(outputs_train_file, batch_size, keep_features=False)
    targets_test = generate_graph(outputs_test_file, batch_size, keep_features=False)

    input_train_ph = generate_placeholder(inputs_train_file, batch_size, keep_features=True)
    input_test_ph = generate_placeholder(inputs_test_file, batch_size, keep_features=True)
    target_train_ph = generate_placeholder(outputs_train_file, batch_size, keep_features=False)
    target_test_ph = generate_placeholder(outputs_test_file, batch_size, keep_features=False)

    output_train = model(input_train_ph, 1)
    output_test = model(input_test_ph, 1)

    loss_train = softmax_loss(target_train_ph, output_train) if use_edges else \
        softmax_loss_on_nodes(target_train_ph, output_train)
    loss_train = sum(loss_train)

    loss_test = softmax_loss(target_test_ph, output_test) if use_edges else \
        softmax_loss_on_nodes(target_test_ph, output_test)
    loss_test = loss_test[-1]
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimize = optimizer.minimize(loss_train)
    saver = tf.train.Saver()

    with tf.device(device):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            last_loss = -1.0

            for iteration in range(1, epochs + 1):

                if iteration > 3:
                    print("{}th epoch".format(iteration))
                elif iteration == 3:
                    print("3rd epoch")
                elif iteration == 2:
                    print("2nd epoch")
                elif iteration == 1:
                    print("1st epoch")

                run_session(sess, inputs_train, input_train_ph, output_train, steps_per_epoch, use_edges=use_edges,
                            targets=targets_train, target_ph=target_train_ph, optimize=optimize, loss=loss_train,
                            print_steps=print_steps)

                current_loss = run_session(sess, inputs_test, input_test_ph, output_test, validation_steps_per_epoch,
                                           use_edges=use_edges, targets=targets_test, target_ph=target_test_ph,
                                           loss=loss_test, print_steps=print_steps)

                if early_stopping:
                    if current_loss < last_loss or last_loss <= 0.0:
                        last_loss = current_loss
                    else:
                        break
            if output_save_path is not None:
                train_len = int(len(open(inputs_train_file).read().split('\n')) / batch_size_)
                test_len = int(len(open(inputs_test_file).read().split('\n')) / batch_size_)
                run_session(sess, inputs_train, input_train_ph, output_train, train_len,
                            output_save_path=output_save_path)
                run_session(sess, inputs_test, input_test_ph, output_test, test_len, output_save_path=output_save_path,
                            accurately=accurately)
            save_path = saver.save(sess, "{}.ckpt".format(save_model_path))
            print("Model saved in path: %s" % save_path)
    return model


def predict(model, checkpoint, data, batch_size, output_save_path, accurately, device='/device:GPU:0'):
    """
    Use the model to predict the output with given input
    :param model: The model used for prediction
    :param checkpoint: The path to the previously saved checkpoint file
    :param data: The path of the input file used for prediction
    :param batch_size: The size of a batch
    :param output_save_path: The path where the final output graphs shall be saved
    :param accurately: Whether to save the scores or just whether the result was one or zero
    :param device: Which device to use. GPU:0 is the default.
    """
    inputs = generate_graph(data, batch_size, keep_features=True)

    input_ph = generate_placeholder(data, batch_size, keep_features=True)

    outputs = model(input_ph, 1)
    saver = tf.train.Saver()
    with tf.device(device):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint = checkpoint if checkpoint.endswith(".ckpt") else "{}.ckpt".format(checkpoint)
            saver.restore(sess, checkpoint)
            steps = int(len(open(data).read().split('\n')) / batch_size_)
            run_session(sess, inputs, input_ph, outputs, steps, output_save_path=output_save_path, accurately=accurately)


def predict_one_graph(model, checkpoint, json_data, device='/device:GPU:0'):
    """
    Predicts one output for the one given input
    :param model: The model used for prediction
    :param checkpoint: The path to the previously saved checkpoint file
    :param json_data:
    :param device: Which device to use. GPU:0 is the default.
    :return: The scored ooutput
    """
    data_dict_list = [process_line(json_dict=json_data, keep_features=True, existence_as_vector=False)]
    inputs = utils_tf.data_dicts_to_graphs_tuple(data_dict_list)
    outputs = model(inputs, 1)
    saver = tf.train.Saver()
    with tf.device(device):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint = checkpoint if checkpoint.endswith(".ckpt") else "{}.ckpt".format(checkpoint)
            saver.restore(sess, checkpoint)
            values = sess.run({"inputs": inputs, "outputs": outputs})
    inputs_dict = utils_np.graphs_tuple_to_data_dicts(values["inputs"])[0]
    outputs_dict = utils_np.graphs_tuple_to_data_dicts(values["outputs"][0])[0]
    return {"nodes": [[i.tolist(), o[1]] for (i, o) in zip(inputs_dict["nodes"], outputs_dict["nodes"])],
            "edges": [[i[0], o[1]] for (i, o) in zip(inputs_dict["edges"], outputs_dict["edges"])],
            "globals": [float(g) for g in inputs_dict["globals"]],
            "senders": inputs_dict["senders"].tolist(),
            "receivers": inputs_dict["receivers"].tolist()}


def test(model, checkpoint, input_data, target_data, batch_size, output_save_path, accurately, use_edges=True,
         device='/device:GPU:0', print_steps=False):
    """
    Tests the model at the checkpoint.
    :param model: The model to test
    :param checkpoint: The path to the previously saved checkpoint file
    :param input_data: The path of the input file used for testing
    :param target_data: The path of the target file used for testing
    :param batch_size: The size of a test batch
    :param output_save_path: The path where the final output graphs shall be saved
    :param accurately: Whether to save the scores or just whether the result was one or zero
    :param use_edges: Whether or not to test on the edges as well as the nodes
    :param device: Which device to use. GPU:0 is the default.
    :param print_steps: Whether to print each batch's report
    """
    inputs = generate_graph(input_data, batch_size, keep_features=True)
    targets = generate_graph(target_data, batch_size, keep_features=False)

    input_ph = generate_placeholder(input_data, batch_size, keep_features=True)
    target_ph = generate_placeholder(target_data, batch_size, keep_features=False)

    outputs = model(input_ph, 1)

    loss = softmax_loss(target_ph, outputs) if use_edges else \
        softmax_loss_on_nodes(target_ph, outputs)
    loss = loss[-1]

    saver = tf.train.Saver()
    with tf.device(device):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            checkpoint = checkpoint if checkpoint.endswith(".ckpt") else "{}.ckpt".format(checkpoint)
            saver.restore(sess, checkpoint)
            steps = int(len(open(input_data).read().split('\n')) / batch_size_)
            run_session(sess, inputs, input_ph, outputs, steps, targets=targets, target_ph=target_ph, loss=loss,
                        use_edges=use_edges, print_steps=print_steps, output_save_path=output_save_path,
                        accurately=accurately)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.reset_default_graph()
    num_processing_steps = 1

    epochs_ = 2
    batch_size_ = 8

    encode_process_decode_model = EncodeProcessDecode(edge_output_size=2, node_output_size=2, global_output_size=1)
    graph_dependent_model = GraphAttention(edge_output_size=2, node_output_size=2, global_output_size=1)

    training_steps = int(len(open('./data/sentences_train2.jsonl').read().split('\n')) / batch_size_)
    validation_steps = int(len(open('./data/sentences_test2.jsonl').read().split('\n')) / batch_size_)

    """train_generator(graph_dependent_model, epochs_, batch_size_, training_steps, validation_steps,
                    './data/sentences_train3.jsonl',
                    './data/highlight_sentences_train3.jsonl',
                    './data/sentences_test3.jsonl',
                    './data/highlight_sentences_test3.jsonl',
                    './data/predictions3.jsonl',
                    './chkpt/model_checkpoint',
                    use_edges=False,
                    print_steps=True,
                    device='/device:CPU:0')"""

    predict(graph_dependent_model, './chkpt4/model_checkpoint', './data/sentences2.jsonl', batch_size_,
            "./data/tmp/pred.jsonl", device='/device:CPU:0')

    """test(graph_dependent_model, './chkpt/model_checkpoint', './data/sentences_test2.jsonl',
         './data/highlight_sentences_test2.jsonl', batch_size_, "./data/tmp/pred2.jsonl", use_edges=False,
         device='/device:CPU:0')"""


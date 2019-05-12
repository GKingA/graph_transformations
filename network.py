from graph_nets import utils_tf
from graph_nets.demos.models import EncodeProcessDecode
import tensorflow as tf

from graph_transformations.models.model_with_attention import GraphAttention
from graph_transformations.graph_losses import softmax_loss
from graph_transformations.graph_file_handling import get_first_batch_graph_dict, generate_graph, save_predicted_graphs
from graph_transformations.compute_measures import compute_accuracy


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
        save_predicted_graphs(output_save_path, train_values["inputs"], train_values["outputs"][-1],
                              test_values["inputs"], test_values["outputs"][-1])


def train_generator(model, epochs, batch_size, steps_per_epoch, validation_steps_per_epoch,
                    inputs_train_file, outputs_train_file,
                    inputs_test_file, outputs_test_file, output_save_path):
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
    :return:
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

    loss_train = softmax_loss(target_train_ph, output_train)
    loss_train = sum(loss_train)

    loss_test = softmax_loss(target_test_ph, output_test)
    loss_test = loss_test[-1]
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimize = optimizer.minimize(loss_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(1, epochs + 1):

            if iteration > 3:
                print("{}th epoch".format(iteration))
            elif iteration == 3:
                print("3rd epoch")
            elif iteration == 2:
                print("2nd epoch")
            elif iteration == 1:
                print("1st epoch")

            losses = []
            corrects = []
            solved = []
            train_values = None
            test_values = None
            step = 0
            for input_batch, output_batch in zip(inputs_train, targets_train):
                feed_dict = {input_train_ph: input_batch, target_train_ph: output_batch}
                train_values = sess.run({
                    "optimize": optimize,
                    "inputs": input_train_ph,
                    "targets": target_train_ph,
                    "loss": loss_train,
                    "outputs": output_train
                }, feed_dict=feed_dict)
                correct_train, solved_train = compute_accuracy(train_values["targets"], train_values["outputs"][-1])
                print("Train loss: {}\tCorrect train parts: {}\tCorrectly solved train graphs: {}".format(
                    train_values["loss"], correct_train, solved_train))
                if step < steps_per_epoch:
                    step += 1
                else:
                    step = 0
                    break
            for input_batch, output_batch in zip(inputs_test, targets_test):
                feed_dict = {input_test_ph: input_batch, target_test_ph: output_batch}
                test_values = sess.run({
                    "inputs": input_test_ph,
                    "targets": target_test_ph,
                    "loss": loss_test,
                    "outputs": output_test,
                }, feed_dict=feed_dict)
                correct_test, solved_test = compute_accuracy(test_values["targets"], test_values["outputs"][-1])
                losses.append(test_values["loss"])
                corrects.append(correct_test)
                solved.append(solved_test)
                if step < validation_steps_per_epoch:
                    step += 1
                else:
                    break
            print("Test loss: {}\tCorrect test parts: {}\tCorrectly solved test graphs: {}".format(
                sum(losses)/validation_steps_per_epoch, sum(corrects)/validation_steps_per_epoch,
                sum(solved)/validation_steps_per_epoch))
        save_predicted_graphs(output_save_path, train_values["inputs"], train_values["outputs"][-1],
                              test_values["inputs"], test_values["outputs"][-1])


if __name__ == '__main__':
    tf.reset_default_graph()
    num_processing_steps = 10

    encode_process_decode_model = EncodeProcessDecode(edge_output_size=2, node_output_size=2, global_output_size=1)
    # graph_dependent_model = GraphAttention(edge_output_size=2, node_output_size=2, global_output_size=1)

    epochs = 10000
    batch_size = 32

    training_steps = int(len(open('graph_transformations/data/sentences_train.jsonl').read().split('\n')) / batch_size)
    validation_steps = int(len(open('graph_transformations/data/sentences_test.jsonl').read().split('\n')) / batch_size)

    train_generator(encode_process_decode_model, epochs, batch_size, training_steps, validation_steps,
                    'graph_transformations/data/sentences_train.jsonl',
                    'graph_transformations/data/highlight_sentences_train.jsonl',
                    'graph_transformations/data/sentences_test.jsonl',
                    'graph_transformations/data/highlight_sentences_test.jsonl',
                    'graph_transformations/data/predictions.jsonl')


from tensorflow._api.v1.losses import mean_pairwise_squared_error as pairwise_mse
from tensorflow._api.v1.losses import sigmoid_cross_entropy, softmax_cross_entropy


def regression_loss(target, outputs):
    """
    Calculates pairwise root mean squared error on nodes and edges
    :param target: The target graph
    :param outputs: List of output graphs
    :return: It returns the calculated loss
    """
    loss_ = [tf.sqrt(pairwise_mse(target.nodes, output.nodes)) + tf.sqrt(pairwise_mse(target.edges, output.edges))
             for output in outputs]
    return loss_


def binary_categorical_loss(target, outputs):
    """
    Calculates the cross entropy loss on nodes and edges
    :param target: The target graph
    :param outputs: List of output graphs
    :return: It returns the calculated loss
    """
    loss_ = [sigmoid_cross_entropy(target.nodes, output.nodes) + sigmoid_cross_entropy(target.edges, output.edges)
             for output in outputs]
    return loss_


def softmax_loss(target, outputs):
    """
    Calculates the categorical cross entropy loss on nodes and edges
    :param target: The target graph
    :param outputs: List of output graphs
    :return: It returns the calculated loss
    """
    loss_ = [softmax_cross_entropy(target.nodes, output.nodes) + softmax_cross_entropy(target.edges, output.edges)
             for output in outputs]
    return loss_

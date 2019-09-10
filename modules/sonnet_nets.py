import sonnet as snt
import tensorflow as tf
import tensorflow_hub as hub


class ActivatedLSTM(snt.AbstractModule):
    def __init__(self, hidden_size, number_of_layers=4, batch_size=32, activation_funcion=tf.nn.relu):
        super(ActivatedLSTM, self).__init__(name="ActivatedLSTM")

        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(hidden_size, activation=activation_funcion)

        self.lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(number_of_layers)])
        self.state = self.lstm.zero_state(batch_size, tf.float32)
        self.activation_function = activation_funcion

    def _build(self, inputs, steps=10):
        for _ in range(steps):
            output, self.state = self.lstm(inputs, self.state)
        return self.activation_function(output)


class ActivatedLinear(snt.AbstractModule):
    def __init__(self, output_size, activation_funcion=tf.nn.relu, **kwargs):
        super(ActivatedLinear, self).__init__(name="ActivatedLinear")
        self.linear = snt.Linear(output_size=output_size, **kwargs)
        self.activation_function = activation_funcion

    def _build(self, inputs):
        return self.activation_function(self.linear(inputs))


class NodeEmbedding(snt.AbstractModule):
    def __init__(self, elmo, name="NodeEmbedding"):
        super(NodeEmbedding, self).__init__(name=name)
        self.word_elmo = elmo #hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        #self.pos_elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    def _build(self, inputs):
        return self.word_elmo(tf.transpose(inputs)[0])  # , self.pos_elmo(tf.transpose(inputs)[1])


class EdgeEmbedding(snt.AbstractModule):
    def __init__(self, elmo, name="EdgeEmbedding"):
        super(EdgeEmbedding, self).__init__(name=name)
        self.elmo = elmo #hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    def _build(self, inputs):
        return self.elmo(tf.transpose(inputs)[0])

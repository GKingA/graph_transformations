import sonnet as snt
import tensorflow as tf


class ActivatedLSTM(snt.AbstractModule):
    def __init__(self, hidden_size, activation_funcion=tf.nn.relu, use_dropout=True, keep_prob=0.75, **kwargs):
        self.lstm = snt.LSTM(hidden_size=hidden_size, **kwargs) if not use_dropout else \
            snt.lstm_with_recurrent_dropout(hidden_size, keep_prob, **kwargs)
        self.activation_function = activation_funcion

    def _build(self, inputs):
        return self.activation_function(self.lstm(inputs))


class ActivatedLinear(snt.AbstractModule):
    def __init__(self, output_size, activation_funcion=tf.nn.relu, **kwargs):
        self.linear = snt.Linear(output_size=output_size, **kwargs)
        self.activation_function = activation_funcion

    def _build(self, inputs):
        return self.activation_function(self.linear(inputs))

from graph_nets import modules as graph_net_modules
from graph_nets import utils_tf
import tensorflow_hub as hub
from graph_transformations.modules import sonnet_nets
import sonnet as snt
import tensorflow as tf

HIDDEN_SIZE = 512


class Encoder(snt.AbstractModule):
    def __init__(self, name="Encoder"):
        super(Encoder, self).__init__(name=name)
        self.word_elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        self.edge_elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        with self._enter_variable_scope():
            self._encode = graph_net_modules.GraphIndependent(edge_model_fn=self.edge_model_fn,
                                                              node_model_fn=self.node_model_fn,
                                                              global_model_fn=self.global_model_fn)

    def _build(self, inputs):
        return self._encode(inputs)

    def node_model_fn(self):
        return sonnet_nets.NodeEmbedding(self.word_elmo)

    def edge_model_fn(self):
        return sonnet_nets.EdgeEmbedding(self.edge_elmo)

    def global_model_fn(self):
        return snt.Linear(1)


class GraphAttention(snt.AbstractModule):
    def __init__(self, edge_output_size=None, node_output_size=None, global_output_size=None,
                 edge_layer_activation=tf.nn.relu, node_layer_activation=tf.nn.relu,
                 global_layer_activation=tf.nn.relu, last_edge_layer_activation=tf.nn.softmax,
                 last_node_layer_activation=tf.nn.softmax, last_global_layer_activation=tf.keras.activations.linear,
                 edge_vocab_size=20, edge_embed_dim=100, node_vocab_size=1000, node_embed_dim=100,
                 name="GraphAttention"):

        super(GraphAttention, self).__init__(name=name)

        self.edge_layer_activation = edge_layer_activation
        self.node_layer_activation = node_layer_activation
        self.global_layer_activation = global_layer_activation
        self.edge_vocab_size = edge_vocab_size
        self.edge_embed_dim = edge_embed_dim
        self.node_vocab_size = node_vocab_size
        self.node_embed_dim = node_embed_dim

        self._encoder = Encoder()

        self._network = graph_net_modules.GraphNetwork(edge_model_fn=self.edge_model_fn,
                                                       node_model_fn=self.node_model_fn,
                                                       global_model_fn=self.global_model_fn,
                                                       reducer=tf.unsorted_segment_sum)

        # Transforms the outputs into the appropriate shapes.
        edge_fn = None if edge_output_size is None else \
            lambda: sonnet_nets.ActivatedLinear(edge_output_size, last_edge_layer_activation)
        node_fn = None if node_output_size is None else \
            lambda: sonnet_nets.ActivatedLinear(node_output_size, last_node_layer_activation)
        global_fn = None if global_output_size is None else \
            lambda: sonnet_nets.ActivatedLinear(global_output_size, last_global_layer_activation)
        with self._enter_variable_scope():
            self._output_transform = graph_net_modules.GraphIndependent(edge_fn, node_fn, global_fn)

    def edge_model_fn(self):
        return snt.Sequential([
            sonnet_nets.ActivatedLSTM(hidden_size=HIDDEN_SIZE, activation_funcion=self.edge_layer_activation),
            sonnet_nets.ActivatedLinear(int(HIDDEN_SIZE / 2), self.edge_layer_activation),
            sonnet_nets.ActivatedLinear(int(HIDDEN_SIZE / 4), self.edge_layer_activation)
        ])

    def node_model_fn(self):
        return snt.Sequential([
            sonnet_nets.ActivatedLSTM(hidden_size=HIDDEN_SIZE, activation_funcion=self.node_layer_activation),
            sonnet_nets.ActivatedLinear(int(HIDDEN_SIZE / 2), self.node_layer_activation),
            sonnet_nets.ActivatedLinear(int(HIDDEN_SIZE / 4), self.node_layer_activation)
        ])

    def global_model_fn(self):
        return snt.Sequential([
            sonnet_nets.ActivatedLinear(16, self.global_layer_activation),
            sonnet_nets.ActivatedLinear(32, self.global_layer_activation),
            sonnet_nets.ActivatedLinear(16, self.global_layer_activation),
            snt.LayerNorm()
        ])

    def _build(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._network(core_input)
            decoded_op = self._output_transform(latent)
            output_ops.append(decoded_op)
        return output_ops

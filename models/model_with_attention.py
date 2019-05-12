from graph_nets import modules as graph_net_modules
from graph_nets import utils_tf
from graph_transformations.modules import sonnet_nets
import sonnet as snt
import tensorflow as tf

HIDDEN_SIZE = 12


class GraphAttention(snt.AbstractModule):
    def __init__(self, edge_output_size=None, node_output_size=None, global_output_size=None,
                 edge_layer_activation=tf.nn.relu, node_layer_activation=tf.nn.relu,
                 global_layer_activation=tf.nn.relu, last_edge_layer_activation=tf.sigmoid,
                 last_node_layer_activation=tf.sigmoid, last_global_layer_activation=tf.keras.activations.linear,
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

        self._network = graph_net_modules.GraphNetwork(edge_model_fn=self.edge_model_fn,
                                                       node_model_fn=self.node_model_fn,
                                                       global_model_fn=self.global_model_fn,
                                                       reducer=tf.unsorted_segment_sum)

        # Transforms the outputs into the appropriate shapes.
        edge_fn = None if edge_output_size is None else \
            lambda: last_edge_layer_activation(snt.Linear(edge_output_size, name="edge_output"))
        node_fn = None if node_output_size is None else \
            lambda: last_node_layer_activation(snt.Linear(node_output_size, name="node_output"))
        global_fn = None if global_output_size is None else \
            lambda: last_global_layer_activation(snt.Linear(global_output_size, name="global_output"))
        with self._enter_variable_scope():
            self._output_transform = graph_net_modules.GraphIndependent(edge_fn, node_fn, global_fn)

    def edge_model_fn(self):
        return snt.Sequential([
            snt.Embed(vocab_size=self.edge_vocab_size, embed_dim=self.edge_embed_dim),
            sonnet_nets.ActivatedLSTM(hidden_size=HIDDEN_SIZE, activation_funcion=self.edge_layer_activation,
                                      forget_bias=0.75),
            graph_net_modules.SelfAttention()
        ])

    def node_model_fn(self):
        return snt.Sequential([
            snt.Embed(vocab_size=self.node_vocab_size, embed_dim=self.node_embed_dim),
            sonnet_nets.ActivatedLSTM(hidden_size=HIDDEN_SIZE, activation_funcion=self.node_layer_activation,
                                      forget_bias=0.75),
            graph_net_modules.SelfAttention()
        ])

    def global_model_fn(self):
        return snt.Sequential([
            sonnet_nets.ActivatedLinear(16, self.global_layer_activation),
            sonnet_nets.ActivatedLinear(32, self.global_layer_activation),
            sonnet_nets.ActivatedLinear(16, self.global_layer_activation),
            snt.LayerNorm()
        ])

    def _build(self, input_op, num_processing_steps):
        input_op0 = input_op
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([input_op0, input_op], axis=1)
            input_op = self._network(core_input)
            output_ops.append(self._output_transform(input_op))
        return output_ops

import sonnet as snt
import tensorflow as tf
from graph_nets import blocks
from graph_nets.modules import _unsorted_segment_softmax, _received_edges_normalizer


class ActivatedLSTM(snt.AbstractModule):
    def __init__(self, hidden_size, number_of_layers=4, batch_size=8, activation_funcion=tf.nn.relu):
        super(ActivatedLSTM, self).__init__(name="ActivatedLSTM")

        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(hidden_size, activation=activation_funcion)

        self.lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(number_of_layers)])
        self.state = self.lstm.zero_state(batch_size, tf.float32)
        self.activation_function = activation_funcion

    def _build(self, inputs):
        outputs, state = tf.nn.dynamic_rnn(cell=self.lstm, inputs=tf.expand_dims(inputs, axis=2), dtype=tf.float32)
        return self.activation_function(outputs)


class ActivatedLinear(snt.AbstractModule):
    def __init__(self, output_size, activation_funcion=tf.nn.relu, **kwargs):
        super(ActivatedLinear, self).__init__(name="ActivatedLinear")
        self.linear = snt.Linear(output_size=output_size, **kwargs)
        self.activation_function = activation_funcion

    def _build(self, inputs):
        if inputs.shape.ndims == 3:
            inputs = tf.reshape(inputs, [-1, inputs.shape.dims[1].value * inputs.shape.dims[2].value])
        return self.activation_function(self.linear(inputs))


class NodeEmbedding(snt.AbstractModule):
    def __init__(self, elmo, name="NodeEmbedding"):
        super(NodeEmbedding, self).__init__(name=name)
        self.word_elmo = elmo

    def _build(self, inputs):
        return self.word_elmo(tf.transpose(inputs)[0])


class EdgeEmbedding(snt.AbstractModule):
    def __init__(self, elmo, name="EdgeEmbedding"):
        super(EdgeEmbedding, self).__init__(name=name)
        self.elmo = elmo

    def _build(self, inputs):
        return self.elmo(tf.transpose(inputs)[0])


class SimplifiedSelfAttention(snt.AbstractModule):
    def __init__(self, name="simplified_self_attention"):
        super(SimplifiedSelfAttention, self).__init__(name=name)
        self._normalizer = _unsorted_segment_softmax

    def _build(self, node_values, node_keys, node_queries, attention_graph):
        # Sender nodes put their keys and values in the edges.
        # [total_num_edges, num_heads, query_size]
        sender_keys = blocks.broadcast_sender_nodes_to_edges(attention_graph.replace(nodes=node_keys))

        # [total_num_edges, num_heads, value_size]
        sender_values = blocks.broadcast_sender_nodes_to_edges(attention_graph.replace(nodes=node_values))

        # Receiver nodes put their queries in the edges.
        # [total_num_edges, num_heads, key_size]
        receiver_queries = blocks.broadcast_receiver_nodes_to_edges(attention_graph.replace(nodes=node_queries))

        # Attention weight for each edge.
        # [total_num_edges, num_heads]
        attention_weights_logits = tf.reduce_sum(sender_keys * tf.transpose(receiver_queries), axis=-1)
        normalized_attention_weights = _received_edges_normalizer(attention_graph.replace(edges=attention_weights_logits), normalizer=self._normalizer)

        # Attending to sender values according to the weights.
        # [total_num_edges, num_heads, embedding_size]
        attented_edges = sender_values * normalized_attention_weights[..., None]

        # Summing all of the attended values from each node.
        # [total_num_nodes, num_heads, embedding_size]
        received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(reducer=tf.unsorted_segment_sum)
        aggregated_attended_values = received_edges_aggregator(attention_graph.replace(edges=attented_edges))

        return attention_graph.replace(nodes=aggregated_attended_values)


class GraphAttentionLayer(snt.AbstractModule):
    def __init__(self, input_size, output_size, heads, name="gat", **kwargs):
        super(GraphAttentionLayer, self).__init__(name=name)
        self.heads = heads
        self.W = []
        self.attentions = []
        for _ in range(heads):
            self.W.append(tf.Variable(tf.random_normal([output_size, input_size]), trainable=True, name="W"))
            self.attentions.append(snt.Linear(output_size=1, **kwargs))

    def _build(self, attended_graph):
        stacked_edges = tf.stack([blocks.broadcast_sender_nodes_to_edges(attended_graph),
                                  blocks.broadcast_receiver_nodes_to_edges(attended_graph)], axis=1)
        his = None
        for k in range(self.heads):
            e = tf.map_fn(lambda edge: tf.concat([tf.tensordot(self.W[k], edge[0], axes=1),
                                                  tf.tensordot(self.W[k], edge[1], axes=1)], axis=0), stacked_edges)
            attended_e = tf.exp(tf.nn.leaky_relu(self.attentions[k](e)))

            e_sender_sum = tf.math.unsorted_segment_sum(attended_e, attended_graph.senders,
                                                        num_segments=tf.shape(attended_graph.nodes)[0])
            e_receiver_sum = tf.math.unsorted_segment_sum(attended_e, attended_graph.receivers,
                                                          num_segments=tf.shape(attended_graph.nodes)[0])
            stacked_to_avg = tf.stack([attended_e, tf.add(tf.gather(e_sender_sum, attended_graph.senders),
                                                          tf.gather(e_receiver_sum, attended_graph.receivers))], axis=1)
            e_avg = tf.map_fn(lambda avg: tf.divide(avg[0], avg[1]), stacked_to_avg)

            Whi = tf.map_fn(lambda edge: tf.tensordot(self.W[k], edge, axes=1),
                            blocks.broadcast_sender_nodes_to_edges(attended_graph))
            aWhi = tf.multiply(Whi, e_avg)
            hi = tf.math.unsorted_segment_sum(aWhi, attended_graph.senders,
                                              num_segments=tf.shape(attended_graph.nodes)[0])
            if his is None:
                his = hi
            else:
                his = tf.add(his, hi)
        his = tf.divide(his, self.heads)
        return attended_graph.replace(nodes=his)

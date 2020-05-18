import tensorflow as tf
from tensorflow.keras.layers import Layer

def _get_segment_indices( ids_):
    head = tf.reverse(tf.add(tf.reduce_max(ids_, axis=0), 1), [0])
    max_columns = tf.concat([tf.reverse(tf.slice(head, [0], \
                            [tf.size(head)-1]), [0]), [1]], 0)
    multipliers = tf.math.cumprod(max_columns, reverse=True)
    y, idx = tf.unique(tf.reduce_sum(tf.multiply(ids_, multipliers), axis=1))
    return idx

def _reduce_indices( ids_):
    ids_shape   = tf.shape(ids_)
    root_shape  = tf.gather(ids_shape, tf.range(0,tf.size(ids_shape)-1))
    last_column = tf.gather(ids_shape, [tf.size(ids_shape)-1])
    new_shape = tf.concat([last_column-1, root_shape], 0)
    reduced_ids = tf.reshape(tf.gather(tf.reshape(tf.transpose(ids_), [-1]),\
                             tf.range(0, tf.reduce_prod(new_shape))), \
                             new_shape)
    reduced_ids = tf.transpose(reduced_ids)
    return reduced_ids

def reduce_segment_ids(segment_ids, n, return_new_indices=False):
    lengths = segment_ids
    for i in range(n):
        segment_indices = _get_segment_indices(lengths)
        reduced_ids = _reduce_indices(lengths)
        lengths = tf.math.segment_max(reduced_ids, segment_indices)
    if return_new_indices:
        return segment_indices, reduced_ids
    return segment_indices

class BagLayer(Layer):
    #@interfaces.legacy_dense_support
    def __init__(self, agg_type='max', **kwargs):
        assert agg_type in ['max', 'sum', 'mean']
        super(BagLayer, self).__init__(**kwargs)
        self.agg_type = agg_type
        if self.agg_type == 'max':
            self.agg_funct = tf.math.segment_max
        elif self.agg_type == 'sum':
            self.agg_funct = tf.math.segment_sum
        elif self.agg_type == 'mean':
            self.agg_funct = tf.math.segment_mean
        else:
            self.agg_funct = None

#    def build(self, input_shape):
#        assert type(input_shape) == list
#        assert len(input_shape) == 2
#        self.w = self.add_weight(shape=(input_shape[0][-1], self.units),
#                             initializer='random_normal',
#                             trainable=True)
#        if self.use_bias:
#            self.b = self.add_weight(shape=(self.units,),
#                                     initializer='random_uniform',
#                                     trainable=True)

    def call(self, inputs):
        assert type(inputs) == list
        assert len(inputs) == 2
        x, s = inputs
        h = self.agg_funct(x, s)
        return h

    def get_config(self):
        config = super(LafLayer, self).get_config()
        config.update({'agg_type':self.agg_type, 'agg_funct':self.agg_funct})
        return config

    def compute_output_shape(self, input_shape):
        assert type(inputs) == list
        assert len(inputs) == 2
        return tuple(None, input_shape[0][-1])

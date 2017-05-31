import tensorflow as tf
from ops import linear, conv2d, flatten
from ops import invariant_layer, mask_and_pool

def deepmind_CNN(state, output_size=128):
    w = {}
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    state = tf.transpose(state, perm=[0, 2, 3, 1])

    l1, w['l1_w'], w['l1_b'] = conv2d(state,
      32, [8, 8], [4, 4], initializer, activation_fn, 'NHWC', name='l1')
    l2, w['l2_w'], w['l2_b'] = conv2d(l1,
      64, [4, 4], [2, 2], initializer, activation_fn, 'NHWC', name='l2')
    l3, w['l3_w'], w['l3_b'] = conv2d(l1, 
      64, [3, 3], [1, 1], initializer, activation_fn, 'NHWC', name='l3')
        
    shape = l3.get_shape().as_list()
    l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

    embedding, w['l4_w'], w['l4_b'] = linear(l3_flat, 128,
      activation_fn=activation_fn, name='value_hid')

    # Returns the network output, parameters
    return embedding, [ v for v in w.values() ]


def feedforward_network(state):
    w = {}
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    l1, w['l1_w'], w['l1_b'] = linear(state, 64,
      activation_fn=activation_fn, name='l1')
    l2, w['l2_w'], w['l2_b'] = linear(state, 64,
      activation_fn=activation_fn, name='l2')

    embedding, w['l3_w'], w['l3_b'] = linear(l2, 128,
      activation_fn=activation_fn, name='value_hid')

    # Returns the network output, parameters
    return embedding, [ v for v in w.values() ]


def embedding_network(state, mask):
    # Placeholder layer sizes
    d_e = [[64], [64, 128]]
    d_o = [64, 64]

    # Build graph:
    initial_elems = state

    # Embedding Part
    for i, block in enumerate(d_e):
        el = initial_elems
        for j, layer in enuerate(block):
            context = c if j==0 and not i==0 else None
            el, _ = invariant_layer(el, layer, context=context, name='l' + str(i) + '_'  + str(j))

        c = mask_and_pool(l, mask) # pool to get context for next block
    
    # Fully connected part
    fc = c
    for i, layer in enumerate(d_o):
        fc, _, _ = linear(fc, layer, name='lO_' + str(i))
    
    # Output
    embedding = fc

    # Returns the network output and parameters
    return embedding, []
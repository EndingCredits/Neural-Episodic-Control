import numpy as np
import tensorflow as tf
from ops import linear, conv2d, flatten
from ops import invariant_layer, relation_layer, mask_and_pool, get_mask


def deepmind_CNN(state, output_size=128, seed=123):
    w = {}
    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.1, seed=seed)
    activation_fn = tf.nn.relu
    
    state = tf.transpose(state, perm=[0, 2, 3, 1])

    state = tf.truediv(state, 255.0) #Should probably be 255, but 256 is more efficient(?)
    l1, w['l1_w'], w['l1_b'] = conv2d(state,
      32, [8, 8], [4, 4], initializer, activation_fn, 'NHWC', name='l1')
    l2, w['l2_w'], w['l2_b'] = conv2d(l1,
      64, [4, 4], [2, 2], initializer, activation_fn, 'NHWC', name='l2')
    l3, w['l3_w'], w['l3_b'] = conv2d(l2, 
      64, [3, 3], [1, 1], initializer, activation_fn, 'NHWC', name='l3')

    shape = l3.get_shape().as_list()
    l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

    #l1, w['l1_w'], w['l1_b'] = conv2d(state/255.,
    #    16, [8, 8], [4, 4], initializer, activation_fn, 'NHWC', name='l1')
    #l2, w['l2_w'], w['l2_b'] = conv2d(l1,
    #    32, [4, 4], [2, 2], initializer, activation_fn, 'NHWC', name='l2')

    #shape = l2.get_shape().as_list()
    #l2_flat = tf.reshape(l2, [-1, reduce(lambda x, y: x * y, shape[1:])])
      
    embedding, w['l4_w'], w['l4_b'] = linear(l3_flat, 128,
      activation_fn=tf.identity, name='value_hid')

    # Returns the network output, parameters
    return embedding, w.values()


def feedforward_network(state, seed=123):
    w = {}
    initializer = tf.truncated_normal_initializer(0, 0.02, seed=seed)
    activation_fn = tf.nn.relu

    l1, w['l1_w'], w['l1_b'] = linear(state, 64,
      activation_fn=activation_fn, name='l1')
    l2, w['l2_w'], w['l2_b'] = linear(state, 64,
      activation_fn=activation_fn, name='l2')

    embedding, w['l3_w'], w['l3_b'] = linear(l2, 128,
      activation_fn=activation_fn, name='value_hid')

    # Returns the network output, parameters
    return embedding, [ v for v in w.values() ]

def object_embedding_network(state, n_actions=8):
    mask = get_mask(state)
    #net = embedding_network(state, mask)
    net = relation_network(state, mask)
    return net#, []
    
def embedding_network(state, mask, seed=123):
    # Placeholder layer sizes
    d_e = [[64], [64, 128]]
    d_o = [128]

    # Build graph:
    initial_elems = state

    # Embedding Part
    for i, block in enumerate(d_e):
        el = initial_elems
        for j, layer in enumerate(block):
            context = c if j==0 and not i==0 else None
            el, _ = invariant_layer(el, layer, context=context, name='l' + str(i) + '_'  + str(j), seed=seed+i+j)

        c = mask_and_pool(el, mask) # pool to get context for next block
    
    # Fully connected part
    fc = c
    for i, layer in enumerate(d_o):
        fc, _, _ = linear(fc, layer, activation_fn=tf.nn.relu, name='lO_' + str(i))
    
    # Output
    embedding = fc

    # Returns the network output and parameters
    return embedding, []
    
    
def relation_network(state, mask, seed=123):
    # Placeholder layer sizes
    d_e = [64, 64, 64]
    d_o = [128, 128]

    # Build graph:
    initial_elems = state

    # Embedding Part
    for i, layer in enumerate(d_e):
        el = initial_elems
        el, _ = relation_layer(layer, el, mask, name='l' + str(i))

    c = mask_and_pool(el, mask) # pool to get context for next block
    
    # Fully connected part
    fc = c
    for i, layer in enumerate(d_o):
        fc, _, _ = linear(fc, layer, name='lO_' + str(i))
    
    # Output
    embedding = fc

    # Returns the network output and parameters
    return embedding, []
    
    
def object_embedding_network2(state, l_e, l_o):
    mask = get_mask(state)

    # Embedding Part
    el = state
    
    el, _ = invariant_layer(el, l_e[0], name='l' + str(0))
    for i, l in enumerate(l_e[1:]):
        el = el - tf.expand_dims(mask_and_pool(el, mask), axis=1)
        el, _ = invariant_layer(el, l, name='l' + str(i+1))
        
    c = mask_and_pool(el, mask)
    
    # Fully connected part
    fc = c
    for i, layer in enumerate(l_o):
        fc, _, _ = linear(fc, layer, activation_fn=tf.nn.relu, name='lO_' + str(i))
    
    return fc, []

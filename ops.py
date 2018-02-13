import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
  with tf.variable_scope(name):
    if data_format == 'NCHW':
      stride = [1, 1, stride[0], stride[1]]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
    elif data_format == 'NHWC':
      stride = [1, stride[0], stride[1], 1]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

    w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
    conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

    b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(conv, b, data_format)

  if activation_fn != None:
    out = activation_fn(out)

  return out, w, b

def linear(input_, output_size, stddev=0.1, bias_start=0.0, activation_fn=None, name='linear'):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))

    out = tf.nn.bias_add(tf.matmul(input_, w), b)

    if activation_fn != None:
      return activation_fn(out), w, b
    else:
      return out, w, b

def flatten(input_):
    in_list = [x for x in input_ if x is not None]
    if type(in_list[0]) is list:
      in_list = [flatten(elem) for elem in in_list ]
    
    return tf.concat([ tf.reshape(elem, [-1]) for elem in in_list], axis=0)



# Permutation invariant layer
def invariant_layer(inputs, out_size, context=None, activation_fct='ReLU', name='', seed=123):

    in_size = inputs.get_shape().as_list()[-1]
    if context is not None:
      context_size = context.get_shape().as_list()[-1]

    with tf.variable_scope(name) as vs:
      w_e = tf.Variable(tf.random_normal((in_size,out_size), stddev=0.1, seed=seed), name='w_e')
      if context is not None:
        w_c = tf.Variable(tf.random_normal((context_size,out_size), stddev=0.1, seed=seed), name='w_c')
      b = tf.Variable(tf.zeros(out_size), name='b')

    if context is not None:
       context_part = tf.expand_dims(tf.matmul(context, w_c), 1)
    else:
       context_part = 0
    
    element_part = tf.nn.conv1d(inputs, [w_e], stride=1, padding="SAME")

    elements = tf.nn.relu(element_part + context_part + b)
    
    params = [w_e, w_c, b] if context is not None else [w_e, b]

    # Returns elements, their invariant and  the weights
    return elements, params
    
    
def relation_layer(out_size, state, mask, name=''):
    in_size = state.get_shape()[2]
    num_elems = tf.shape(state)[1]
    
    flat_shape = [-1, num_elems*num_elems, out_size]
    mat_shape = [-1, num_elems, num_elems, out_size]
    
    combined, params_1 = invariant_layer(state, 2*out_size)
    q, k = tf.split(combined, [out_size, out_size], axis=2)
    
    qk = tf.expand_dims(q, -3) + tf.expand_dims(k, -2)
    #qk = tf.reshape(qk, flat_shape)
    #qk = tf.reshape(qk, mat_shape)
    
    mask_ = tf.expand_dims(mask, -3) * tf.expand_dims(mask, -2)
    qk_ = qk - (1-mask_)*10e9
    
    qk = tf.nn.softmax(qk_, dim=3)

    out = tf.reduce_max(qk, 2)# / tf.reduce_sum(mask_, -2)
    
    return out, params_1


def get_mask(x):
    # Returns a matrix with values set to 1 where elements aren't padding
    emb_sum = tf.reduce_sum(tf.abs(x), axis=-1, keep_dims=True)
    return 1.0 - tf.to_float(tf.equal(emb_sum, 0.0))
    

def mask_and_pool(embeds, mask, pool_type='max'):
    # Use broadcasting to multiply
    masked_embeds = tf.multiply(embeds, mask)

    # Pool using max pooling
    embed = tf.reduce_max(masked_embeds, 1)

    # For mean pooling:
    #embed = tf.reduce_sum(masked_embeds, 1) / tf.reduce_sum(mask, 1)

    return embed

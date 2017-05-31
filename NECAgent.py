from __future__ import division

import numpy as np
import tensorflow as tf

import knn_dictionary
from ops import linear, conv2d, flatten

class NECAgent():
    def __init__(self, session, args):

        self.n_input = args.input_size     # Number of features in each observation
        self.n_actions = args.num_actions  # Number of output q_values

        self.discount = args.discount      # Discount factor
        self.n_steps = args.n_step         # Number of steps before truncation of returns
        self.initial_epsilon = args.epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_final = args.epsilon_final
        self.epsilon_anneal = args.epsilon_anneal

        self.DND_size = args.memory_size
        self.delta = args.delta
        self.alpha = args.alpha
        self.number_nn = args.num_neighbours

        self.learning_rate = args.learning_rate
        self.memory_size = args.replay_memory_size
        self.batch_size = args.batch_size
        self.learn_step = args.learn_step

        self.step = 0
        self.seed = 123
        self.curr_seed = self.seed
        self.session = session


        self.memory = ReplayMemory(self.memory_size)

        # Tensorflow variables:

        # Model for Embeddings
        self.state = tf.placeholder("float", [None, 4, 84, 84])
        with tf.variable_scope('embedding'):
            self.state_embeddings, self.weights = self.CNN(self.state)

        # DNDs
        self.DND = knn_dictionary.q_dictionary(self.DND_size, self.state_embeddings.get_shape()[-1], self.n_actions, self.delta, self.alpha)

        self.action = tf.placeholder(tf.int8, [None])

        # Retrieve info from DND dictionary
        embs_and_values = tf.py_func(self.DND._query, [self.state_embeddings, self.action, self.number_nn], [tf.float64, tf.float64])
        self.dnd_embeddings = tf.to_float(embs_and_values[0])
        self.dnd_values = tf.to_float(embs_and_values[1]) 

        # DND calculation
        weightings = 1.0 / (tf.reduce_sum(tf.square(self.dnd_embeddings - tf.expand_dims(self.state_embeddings,1)), axis=2) + [self.delta]) 
        normalised_weightings = weightings / tf.reduce_sum(weightings, axis=1, keep_dims=True) #keep dims for broadcasting
        self.pred_q = tf.reduce_sum(self.dnd_values * normalised_weightings, axis=1)
        
        # Loss Function
        self.target_q = tf.placeholder("float", [None])
        self.td_err = self.target_q - self.pred_q
        total_loss = tf.reduce_sum(tf.square(self.td_err))
        
        # These are the optimiser settings used by DeepMind
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, epsilon=0.01).minimize(total_loss)


    def _get_state_embeddings(self, states):
        # Returns the DND hashes for the given states
        embeddings = self.session.run(self.state_embeddings, feed_dict={self.state: states})
        return embeddings


    def _predict(self, embedding):
        # Return action values for given embedding

        # calculate Q-values
        qs = []
        for a in xrange(self.n_actions):
            if self.DND.dicts[a].queryable(self.number_nn):
                q = self.session.run(self.pred_q, feed_dict={
                    self.state_embeddings: [embedding], self.action: [a]})[0]
            else:
                q = 0.0
            qs.append(q)

        # Return Q values
        return qs


    def _train(self, states, actions, Q_targets):

        if not self.DND.queryable(self.number_nn):
            return False

        self.session.run(self.optim, feed_dict={self.state: states, self.target_q: Q_targets, self.action: actions})
        
        return True


    def Reset(self, state, train=True):
        self.training = train

        #TODO: turn these lists into a proper trajectory object
        self.trajectory_states = [state]
        self.trajectory_embeddings = []
        self.trajectory_values = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_t = 0
        return True

    def GetAction(self):
        state = self.trajectory_states[-1]

        # Get state embedding
        embedding = self._get_state_embeddings([state])[0]

        # get value
        Qs = self._predict(embedding)
        action = np.argmax(Qs) ; value = Qs[action]

        # get action via epsilon-greedy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            value = Qs[action]

        self.trajectory_embeddings.append(embedding)
        self.trajectory_values.append(value)
        return action, value


    def Update(self, action, reward, state, terminal=False):

        self.trajectory_actions.append(action)
        self.trajectory_rewards.append(reward)
        self.trajectory_t += 1
        self.trajectory_states.append(state)

        self.step += 1

        if self.training:

            # Update Epsilon
            per = min(self.step / self.epsilon_anneal, 1)
            self.epsilon = (1-per)*self.initial_epsilon + per*self.epsilon_final

            if self.memory.count > self.batch_size*2 and (self.step % self.learn_step) == 0:
                # Get transition sample from memory
                s, a, R = self.memory.sample(self.batch_size)
                # Run optimization op (backprop)
                self._train(s, a, R)


            # Add to replay memory and DND
            if terminal:
                # Calculate returns
                returns = []
                for t in xrange(self.trajectory_t):
                    if self.trajectory_t - t > self.n_steps:
                        #Get truncated return
                        start_t = t + self.n_steps
                        R_t = self.trajectory_values[start_t]
                    else:
                        start_t = self.trajectory_t
                        R_t = 0
                        
                    for i in xrange(start_t-1, t, -1):
                        R_t = R_t * self.discount + self.trajectory_rewards[i]
                    returns.append(R_t)
                    self.memory.add(self.trajectory_states[t], self.trajectory_actions[t], R_t)

                self.DND.add(self.trajectory_embeddings, self.trajectory_actions, returns)
        return True


    def CNN(self, state):
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

        embedding, w['l4_w'], w['l4_b'] = linear(l3_flat, 128, activation_fn=activation_fn, name='value_hid')

        # Returns the network output, parameters
        return embedding, [ v for v in w.values() ]


    def get_seed(self):
        self.curr_seed += 1
        return self.curr_seed


# Adapted from github.com/devsisters/DQN-tensorflow/
class ReplayMemory:
  def __init__(self, memory_size):
    self.memory_size = memory_size

    self.states = [None]*self.memory_size
    self.actions = np.empty(self.memory_size, dtype=np.int16)
    self.returns = np.empty(self.memory_size, dtype = np.float16)

    self.count = 0
    self.current = 0

  def add(self, state, action, returns):
    self.states[self.current] = state
    self.actions[self.current] = action
    self.returns[self.current] = returns

    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def sample(self, batch_size, seq_len=1):
    # sample random indexes
    indexes = [] ; states_ = []
    watchdog = 0
    while len(indexes) < batch_size:
      # find random index 
      index = np.random.randint(1, self.count - 1)
      indexes.append(index)
      states_.append(self.states[index])

    return states_, self.actions[indexes], self.returns[indexes]

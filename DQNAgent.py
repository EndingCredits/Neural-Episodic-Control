from __future__ import division

import numpy as np
import tensorflow as tf
import scipy#.misc.imresize
#import cv2

from ops import linear

import knn_dictionary


class DQNAgent():
    def __init__(self, session, args):

        # Environment details
        self.obs_size = args.obs_size
        self.n_actions = args.num_actions
        self.viewer = None

        # Agent parameters
        self.discount = args.discount
        self.n_steps = args.n_step
        self.initial_epsilon = args.epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_final = args.epsilon_final
        self.epsilon_anneal = args.epsilon_anneal

        # Training parameters
        self.model_type = args.model
        self.history_len = args.history_len
        self.memory_size = args.replay_memory_size
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.learn_step = args.learn_step
        
        self.name = "Agent"

        # Stored variables
        self.step = 0
        self.started_training = False
        self.seed = args.seed
        self.rng = np.random.RandomState(self.seed)
        self.session = session

        # Replay Memory
        self.memory = ReplayMemory(self.memory_size, self.obs_size)

        # Preprocessor:
        if args.preprocessor == 'deepmind':
            self.preproc = deepmind_preprocessor
        elif args.preprocessor == 'grayscale':
            #incorrect spelling in order to not confuse those silly americans
            self.preproc = greyscale_preprocessor
        else:
            self.preproc = default_preprocessor
            #a lambda could be used here, but I think this makes more sense
        

        # Tensorflow variables:

        # Model for Embeddings
        if self.model_type == 'CNN':
            from networks import deepmind_CNN
            state_dim = [None, self.history_len] + self.obs_size
            model = deepmind_CNN
        elif self.model_type == 'nn':
            from networks import feedforward_network
            state_dim = [None] + self.obs_size
            model = feedforward_network
        elif self.model_type == 'object':
            from networks import object_embedding_network2
            state_dim = [None] + self.obs_size
            model = lambda x: object_embedding_network2( x, args.emb_layers, args.out_layers)
            
            
        self.state = tf.placeholder("float", state_dim)
            
        with tf.variable_scope(self.name + '_pred'):
            emb, _ = model(self.state)
            self.pred_qs, _, _ = linear(tf.nn.relu(emb), self.n_actions)
        with tf.variable_scope(self.name + '_target', reuse=False):
            emb, _ = model(self.state)
            self.target_pred_qs, _, _ = linear(tf.nn.relu(emb), self.n_actions)
            
        self.pred_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_pred')
        self.targ_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'_target') 

        self.action = tf.placeholder('int64', [None])
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0)
        q_acted = tf.reduce_sum(self.pred_qs * action_one_hot, axis=1)
        self.pred_q = q_acted
        
        # Loss Function
        self.target_q = tf.placeholder("float", [None])
        self.td_err = self.target_q - self.pred_q
        total_loss = tf.reduce_sum(tf.square(self.td_err))
        #total_loss = total_loss + become_skynet_penalty #commenting this out makes code run faster 
        
        # Optimiser
        self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate, decay=0.9, epsilon=0.01).minimize(total_loss)
          # These are the optimiser settings used by DeepMind


    def _get_state(self, t=-1):
        # Returns the compiled state from stored observations
        if t==-1: t = self.trajectory_t-1

        if self.history_len == 0:
            state = self.trajectory_observations[t]
        else:
            if self.obs_size[0] == None:
                state = []
                for i in range(self.history_len):
                    state.append(self.trajectory_observations[t-i])
            else:
                state = np.zeros([self.history_len]+self.obs_size)
                for i in range(self.history_len):
                  if (t-i) >= 0:
                    state[i] = self.trajectory_observations[t-i]
        return state


    def _predict(self, state):
        # calculate Q-values
        qs = self.session.run(self.pred_qs, feed_dict={
                  self.state: [state]})[0]
        
        # Return Q values
        return qs
        
    def _eval(self, states):
        # calculate Q-values
        qs = self.session.run(self.target_pred_qs, feed_dict={
                  self.state: states})
        
        # Return Q values
        return np.max(qs, axis=1)


    def _train(self, states, actions, rewards, poststates, terminals):

        self.started_training = True
        
        if self.obs_size[0] == None:
            states, _ = batch_objects(states)
            poststates, _ = batch_objects(poststates)
            
        if False:
            #Predict action with current network
            action = np.argmax(self.pred_qs.eval({self.state: states}), axis=1)
            action_one_hot = np.eye(self.n_actions)[action] #neat little trick for getting one-hot

            # Get value of action from target network
            V_t1 = self.target_pred_qs.eval({self.state: states})
        else:
            V_t1 = self._eval(poststates)
        V_t1 = np.multiply(np.ones(np.shape(terminals)) - terminals, V_t1)
        
        Q_targets = self.discount * V_t1 + rewards

        feed_dict = {
          self.state: states,
          self.target_q: Q_targets,
          self.action: actions
        }

        self.session.run(self.optim, feed_dict=feed_dict)
        
        return True


    def Reset(self, obs, train=True):
        self.training = train

        #TODO: turn these lists into a proper trajectory object
        self.trajectory_observations = [self.preproc(obs)]
        self.trajectory_values = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_t = 0
        return True


    def GetAction(self):
        # TODO: Perform calculations on Update, then use aaved values to select actions
        
        # Get state embedding of last stored state
        state = self._get_state()

        # Get Q-values
        Qs = self._predict(state)
        action = np.argmax(Qs)
        
        #targ_Q = self._eval(state)
        value = Qs[action]#targ_Q

        # Get action via epsilon-greedy
        if True: #self.training:
          if self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.n_actions)
            #value = Qs[action] # Paper uses maxQ, uncomment for on-policy updates

        self.trajectory_values.append(value)
        return action, value


    def Update(self, action, reward, obs, terminal=False):

        self.trajectory_actions.append(action)
        self.trajectory_rewards.append(reward)
        self.trajectory_t += 1
        self.trajectory_observations.append(self.preproc(obs))

        self.step += 1

        if self.training:

            # Update Epsilon
            per = min(self.step / self.epsilon_anneal, 1)
            self.epsilon = (1-per)*self.initial_epsilon + per*self.epsilon_final

            if self.memory.count > self.batch_size*2 and (self.step % self.learn_step) == 0:
                # Get transition sample from memory
                s, a, R, s_, t = self.memory.sample(self.batch_size, self.history_len)
                # Run optimization op (backprop)
                self._train(s, a, R, s_, t)


            # Add to replay memory and DND
            if terminal:
              for t in xrange(self.trajectory_t):
                self.memory.add(self.trajectory_observations[t], self.trajectory_actions[t], self.trajectory_rewards[t], (t==(self.trajectory_t-1)))
                    
            if self.step % 1000 == 0:
                ops = [ self.targ_weights[i].assign(self.pred_weights[i]) for i in range(len(self.targ_weights))]
                self.session.run(ops)
        return True


def batch_objects(input_list):
    # Takes an input list of lists (of vectors), pads each list the length of the longest list,
    #   compiles the list into a single n x m x d array, and returns a corresponding n x m x 1 mask.
    max_len = 0
    out = []; masks = []
    for i in input_list: max_len = max(len(i),max_len)
    for l in input_list:
        # Zero pad output
        out.append(np.pad(np.array(l,dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
        # Create mask...
        masks.append(np.pad(np.array(np.ones((len(l),1)),dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
    return out, masks


# Adapted from github.com/devsisters/DQN-tensorflow/
class ReplayMemory:
  def __init__(self, memory_size, obs_size):
    self.memory_size = memory_size
    self.obs_size = obs_size

    if self.obs_size[0] == None:
        self.observations = [None]*self.memory_size
    else:
        self.observations = np.empty([self.memory_size]+self.obs_size, dtype = np.float16)
    self.actions = np.empty(self.memory_size, dtype=np.int16)
    self.rewards = np.empty(self.memory_size, dtype = np.float16)
    self.terminal = np.empty(self.memory_size, dtype = np.bool_)

    self.count = 0
    self.current = 0

  def add(self, obs, action, rewards, terminal):
    self.observations[self.current] = obs
    self.actions[self.current] = action
    self.rewards[self.current] = rewards
    self.terminal[self.current] = terminal

    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def _get_state(self, index, seq_len):
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    if seq_len == 0:
      state = self.observations[index]
    else:
      if self.obs_size[0] == None:
        state = []
        for i in range(seq_len):
          state.append(self.observations[index-i])
      else:
        state = np.zeros([seq_len]+self.obs_size)
        for i in range(seq_len):
          state[i] = self.observations[index-i]
    return state

  def _uninterrupted(self, start, final):
    if self.current in range(start+1, final):
        return False
    for i in range(start, final-1):
        if self.terminal[i] == True: return False
    return True

  def sample(self, batch_size, seq_len=0):
    # sample random indexes
    indexes = [] ; prestates = [] ; poststates = []
    watchdog = 0
    while len(indexes) < batch_size:
      while True:
        # find random index
        index = np.random.randint(1, self.count - 1)
        if seq_len is not 0:
          start = index-seq_len
          if not self._uninterrupted(start, index+1):
            continue
        break

      indexes.append(index)
      prestates.append(self._get_state(index, seq_len))
      poststates.append(self._get_state(index+1, seq_len))
      
    indexes = np.array(indexes)
    return prestates, self.actions[indexes], self.rewards[indexes], poststates, self.terminal[indexes+1]

# Preprocessors:
def default_preprocessor(state):
    return state

def greyscale_preprocessor(state):
    #state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)/255.
    state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
    return state

def deepmind_preprocessor(state):
    state = greyscale_preprocessor(state)
    #state = np.array(cv2.resize(state, (84, 84)))
    resized_screen = scipy.misc.imresize(state, (110,84))
    state = resized_screen[18:102, :]
    return state


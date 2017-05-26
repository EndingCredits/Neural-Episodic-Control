from __future__ import division

import argparse
import os
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import NECAgent
from environment import ALEEnvironment


def main(_):
  # Set precision for printing numpy arrays, useful for debugging
  #np.set_printoptions(threshold='nan', precision=3, suppress=True)

  # Launch the graph
  with tf.Session() as sess:

    training_iters = args.training_iters
    display_step = args.display_step
    
    env = ALEEnvironment(args.rom)
    args.input_size = 8
    args.num_actions = env.numActions()


    agent = NECAgent.NECAgent(sess, args)

    # Load saver after agent tf variables initialised
    saver = tf.train.Saver()


    # Training, act and learn

    # Initialize the variables
    sess.run(tf.initialize_all_variables())

    # Start Agent
    state = env.reset()
    agent.Reset(state)
    rewards = []

    # Stats for display
    ep_rewards = [] ; ep_reward_last = 0
    qs = [] ; q_last = 0
    avr_ep_reward = max_ep_reward = avr_q = 0.0

    # Keep training until reach max iterations
    for step in tqdm(range(training_iters), ncols=70):

        #env.render()

        # Act, and add 
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)

        # Bookeeping
        rewards.append(reward)
        qs.append(value)

        if terminal:
            #tqdm.write("Resetting...")
            # Bookeeping
            ep_rewards.append(np.sum(rewards))
            rewards = []

            # Reset environment
            state = env.reset()
            agent.Reset(state)


        # Display Statistics
        if (step) % display_step == 0:
            num_eps = len(ep_rewards[ep_reward_last:])
            if num_eps is not 0:
                avr_ep_reward = np.mean(ep_rewards[ep_reward_last:])
                max_ep_reward = np.max(ep_rewards[ep_reward_last:])
                avr_q = np.mean(qs[q_last:]) ; q_last = len(qs)
                ep_reward_last = len(ep_rewards)
            tqdm.write("{}, {:>7}/{}it | {:3n} episodes, q: {:4.3f}, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}, epsilon: {:4.3f}"\
                .format(time.strftime("%H:%M:%S"), step, training_iters, num_eps, avr_q, avr_ep_reward, max_ep_reward, agent.epsilon))
                 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rom', type=str, default='roms/pong.bin',
                       help='Location of rom file')

    parser.add_argument('--training_iters', type=int, default=5000000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=25000,
                       help='Number of iterations between parameter prints')

    parser.add_argument('--learning_rate', type=float, default=0.00001,
                       help='Learning rate for TD updates')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Size of batch for Q-value updates')
    parser.add_argument('--replay_memory_size', type=int, default=50000,
                       help='Size of replay memory')
    parser.add_argument('--learn_step', type=int, default=4,
                       help='Number of steps in between learning updates')

    parser.add_argument('--memory_size', type=int, default=50000,
                       help='Size of DND dictionary')
    parser.add_argument('--num_neighbours', type=int, default=50,
                       help='Number of nearest neighbours to sample from the DND each time')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Alpha parameter for updating stored values')
    parser.add_argument('--delta', type=float, default=0.001,
                       help='Delta parameter for thresholding closeness of neighbours')

    parser.add_argument('--n_step', type=int, default=100,
                       help='Initial epsilon')
    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Initial epsilon')
    parser.add_argument('--epsilon_final', type=float, default=None,
                       help='Final epsilon')
    parser.add_argument('--epsilon_anneal', type=int, default=None,
                       help='Epsilon anneal steps')


    parser.add_argument('--layer_sizes', type=str, default='64',
                       help='Hidden layer sizes for network, separate with comma')

    args = parser.parse_args()

    if args.epsilon_final == None: args.epsilon_final = args.epsilon
    if args.epsilon_anneal == None: args.epsilon_anneal = args.training_iters

    args.layer_sizes = [int(i) for i in (args.layer_sizes.split(',') if args.layer_sizes else [])]

    print args

    tf.app.run()


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

  # Launch the graph
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  with tf.Session(config=config) as sess:

    # Set up training variables
    training_iters = args.training_iters
    display_step = args.display_step
    test_step = args.test_step
    test_count = args.test_count
    tests_done = 0
    test_results = []

    # Stats for display
    ep_rewards = [] ; ep_reward_last = 0
    qs = [] ; q_last = 0
    avr_ep_reward = max_ep_reward = avr_q = 0.0
    # Set precision for printing numpy arrays, useful for debugging
    #np.set_printoptions(threshold='nan', precision=3, suppress=True)

    # Create environment
    #TODO: Determine, model, obs_size, and num_actions automatically from environment 
    if args.env_type == 'ALE':
        env = ALEEnvironment(args.rom)
        # Always use CNN model
        args.model = 'CNN'
        args.preprocessor = 'deepmind'
        args.obs_size = [84,84]
        args.history_len = 4
        args.num_actions = env.numActions()

    elif args.env_type == 'gym':
        import gym
        #import gym_vgdl #This can be found on my github if you want to use it.
        env = gym.make(args.env)
        # Only use CNN model for now
        args.model = 'CNN'
        args.preprocessor = 'grayscale'
        args.obs_size = list(env.observation_space.shape)[:2]
        args.history_len = 2
        args.num_actions = env.action_space.n #only works with discrete action spaces

    # Create agent
    agent = NECAgent.NECAgent(sess, args)

    # Initialize all tensorflow variables
    sess.run(tf.initialize_all_variables())

    # Keep training until reach max iterations

    # Start Agent
    state = env.reset()
    agent.Reset(state)
    rewards = []

    for step in tqdm(range(training_iters), ncols=80):

        #env.render()

        # Act, and add 
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)

        # Bookeeping
        rewards.append(reward)
        qs.append(value)

        if terminal:
            # Bookeeping
            ep_rewards.append(np.sum(rewards))
            rewards = []

            if step >= (tests_done)*test_step:
                R_s = []
                for i in tqdm(range(test_count), ncols=50, bar_format='Testing: |{bar}| {n_fmt}/{total_fmt}'):
                    R = test_agent(agent, env)
                    R_s.append(R)
                tqdm.write("Tests: {}".format(R_s))
                tests_done += 1
                test_results.append({ 'step': step, 'scores': R_s, 'average': np.mean(R_s), 'max': np.max(R_s) })

                #Save to file
                summary = { 'params': args, 'tests': test_results }
                np.save("results.npy", summary)

            # Reset agent and environment
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
            dict_entries = agent.DND.tot_capacity()
            tqdm.write("{}, {:>7}/{}it | {:3n} episodes,"\
                .format(time.strftime("%H:%M:%S"), step, training_iters, num_eps)
                +"q: {:4.3f}, avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}, epsilon: {:4.3f}, entries: {}"\
                .format(avr_q, avr_ep_reward, max_ep_reward, agent.epsilon, dict_entries))
                 

def test_agent(agent, env):
    try:
        state = env.reset(train=False)
    except:
        state = env.reset()
    agent.Reset(state, train=False)
    R = 0

    terminal = False
    while not terminal:
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)
        R += reward
    return R



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rom', type=str, default='roms/pong.bin',
                       help='Location of rom file')
    parser.add_argument('--env', type=str, default='Pong-v0',
                       help='Gym environment to use')
    parser.add_argument('--env_type', type=str, default='ALE',
                       help='Environment to use (ALE or gym)')

    parser.add_argument('--training_iters', type=int, default=5000000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=25000,
                       help='Number of iterations between parameter prints')
    parser.add_argument('--test_step', type=int, default=50000,
                       help='Number of iterations between tests')
    parser.add_argument('--test_count', type=int, default=5,
                       help='Number of test episodes per test')

    parser.add_argument('--learning_rate', type=float, default=0.00001,
                       help='Learning rate for TD updates')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Size of batch for Q-value updates')
    parser.add_argument('--replay_memory_size', type=int, default=100000,
                       help='Size of replay memory')
    parser.add_argument('--learn_step', type=int, default=4,
                       help='Number of steps in between learning updates')

    parser.add_argument('--memory_size', type=int, default=500000,
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


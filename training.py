from __future__ import division

import os
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import NECAgent

#TODO: Split this into a separate agent initiation of agent and env and training
def run_agent(args):
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
    
    mode = args.model
    # Create environment
    if args.env_type == 'ALE':
        from environment import ALEEnvironment
        env = ALEEnvironment(args.rom)
        if mode is None: mode = 'DQN'
        args.num_actions = env.numActions()

    elif args.env_type == 'gym':
        import gym
        try:
            import gym_vgdl #This can be found on my github if you want to use it.
        except:
            pass
        env = gym.make(args.env)
        if mode is None:
            shape = env.observation_space.shape
            if len(shape) is 3: mode = 'DQN'
            elif shape[0] is None: mode = 'object'
            else: mode = 'vanilla'
        args.num_actions = env.action_space.n #only works with discrete action spaces

    # Set agent variables
    if mode=='DQN':
        args.model = 'CNN'
        args.preprocessor = 'deepmind'
        args.obs_size = [84,84]
        args.history_len = 4
    elif mode=='image':
        args.model = 'CNN'
        args.preprocessor = 'grayscale'
        args.obs_size = list(env.observation_space.shape)[:2]
        args.history_len = 2
    elif mode=='object':
        args.model = 'object'
        args.preprocessor = 'default'
        args.obs_size = list(env.observation_space.shape)
        args.history_len = 0
    elif mode=='vanilla':
        args.model = 'nn'
        args.preprocessor = 'default'
        args.obs_size = list(env.observation_space.shape)
        args.history_len = 0

    # Create agent
    agent = NECAgent.NECAgent(sess, args)

    # Initialize all tensorflow variables
    sess.run(tf.global_variables_initializer())


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

            if step >= (tests_done+1)*test_step:
                R_s = []
                for i in tqdm(range(test_count), ncols=50, bar_format='Testing: |{bar}| {n_fmt}/{total_fmt}'):
                    R = test_agent(agent, env)
                    R_s.append(R)
                tqdm.write("Tests: {}".format(R_s))
                tests_done += 1
                test_results.append({ 'step': step, 'scores': R_s, 'average': np.mean(R_s), 'max': np.max(R_s) })

                #Save to file
                summary = { 'params': args, 'tests': test_results }
                if args.save_file is not None:
                    np.save(args.save_file, summary)

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
    #TODO: Add some stochasticity to this somehow so it doesn't just do the same deterministic run 5 times.
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




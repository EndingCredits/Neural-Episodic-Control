import training
import tensorflow as tf

agent_types = ['image', 'objects', 'features']
model_names = { 'image' : 'image', 'objects': 'object', 'features': 'vanilla' }

games = ['aliens', 'boulderdash', 'survivezombies']
suffixes = { 'image' : '', 'objects': '_objects', 'features': '_features' }

class defaults():
    env_type='gym'
    training_iters=1250000
    display_step=25000
    test_step=125000
    test_count=50
    
    learning_rate=0.00001
    batch_size=32
    replay_memory_size=100000
    learn_step=4
    memory_size=500000
    num_neighbours=50
    alpha=0.25
    delta=0.001

    n_step=100
    discount=0.99
    epsilon=0.1
    epsilon_final=0.1
    epsilon_anneal=500000

for g in games:
  for a in agent_types:
    args = defaults()
    args.seed = 123
    args.save_file = a + '_' + g
    args.env = 'vgdl_' + g + suffixes[a] + '-v0'
    args.model = model_names[a]
    training.run_agent(args)
    tf.reset_default_graph()

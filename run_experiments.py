import main
import tensorflow as tf

agent_types = ['image', 'objects', 'features']
games = ['aliens', 'boulderdash', 'survivezombies']

model_names = { 'image' : 'image', 'objects': 'object', 'features': 'vanilla' }
suffixes = { 'image' : '', 'objects': '_objects', 'features': '_features' }

class defaults():
    env_type='gym'
    training_iters=1000000
    display_step=25000
    test_step=50000
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
    epsilon=0.99
    epsilon_final=0.1
    epsilon_anneal=50000

for g in games:
  for a in agent_types:
    args = defaults()
    args.seed = 1234567
    args.save_file = 'results/' + g + '_' + a + '_' + str(args.seed)
    args.env = 'vgdl_' + g + suffixes[a] + '-v0'
    args.model = model_names[a]
    main.run_agent(args)
    tf.reset_default_graph()

import main
import tensorflow as tf

import random

agent_types = ['objects']#['image', 'objects', 'features']
games = ['aliens', 'boulderdash']#['aliens', 'boulderdash', 'survivezombies']

model_names = { 'image' : 'image', 'objects': 'object', 'features': 'vanilla' }
suffixes = { 'image' : '', 'objects': '_objects', 'features': '_features' }

# Learning rate 0.01, 0.003, 0.001, 0.0003
# layer sizes 32, 64, 128, 256
# Num emb layers 2,3,4,5
# Num out layers 2,3,4,5
# Initialisation 0.3, 0.1, 0.03, 0.01
# Replay mem size 100000
# Normalisation for x and y pos

lr = [ 0.01, 0.003, 0.001, 0.0003, 0.0001 ]
ls = [ 32, 64, 128, 256 ]
num_e_l = [ 2, 3, 4, 5 ]
num_o_l = [ 2, 3, 4, 5 ]
init = [ 0.3, 0.1, 0.03, 0.01 ]

class defaults():
    env_type='gym'
    training_iters=2500000
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
    
    args.learning_rate = random.choice(lr)
    layer_size = random.choice(ls)
    num_emb_layers = random.choice(num_e_l)
    num_out_layers = random.choice(num_o_l)
    
    name = "lr{}_ls{}_el{}_ol{}".format(args.learning_rate, layer_size, num_emb_layers, num_out_layers)
    args.save_file = 'hparamres/' + g + '_' + name
    
    args.emb_layers = [layer_size]*num_emb_layers
    args.out_layers = [layer_size]*num_out_layers
    
    
    #args.save_file = 'results/' + g + '_' + n + '_' + str(args.seed)
    args.env = 'vgdl_' + g + suffixes[a] + '-v0'
    args.model = model_names[a]
    print args
    main.run_agent(args)
    tf.reset_default_graph()

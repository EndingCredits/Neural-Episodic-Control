# Neural Episodic Control

This is my attempt at replicating DeepMind's [Neural Episodic Control](https://arxiv.org/abs/1703.01988) agent. It is currently set up for running with the ALE, but can easily be adapted for other environments (you may want to use my older implementation [here](https://github.com/EndingCredits/nn_q_learning_tensorflow/blob/master/NEC.py) as a reference).

To run the code (after installing all dependencies:
```bash
python NEC.py --rom [path/to/rom/file.bin]
```

Further options can be found using:
```bash
python NEC.py -h
```

There is currently only training, without any testing and saving or loading. Scores are reported per episode, which is once per life.

N.B: There are a number of differences between this implementation and the original paper:
* The DND returns exact nearest neighbours and hence is 1/10th of the size used in the paper.
* New elements are checked against previous elements by looking to see if they are closer than a certain threshold. In the NEC paper apparently this is done instead by storing a hash of the game screen and checking for exact matches.
* The way the environment handles new starts is slightly different.
* Various hyperparams may be slightly different.


Many thanks to all the authors whose code I've shamelessly ripped off, e.g. the knn-dictionary code and the environment wrapper (even though now they are probably unrecognisable). If you have a separate working implementation of NEC, I'd love to swap notes to see if I've made any errors or there are any good efficiency savings. Also, if you spot any (inevitable) bugs, please let me know.


## Dependencies

You'll have to look up how to install these, but this project uses the following libraries:
* numpy
* scikit-learn
* tensorflow >1.0
* OpenCV2
* https://github.com/mgbellemare/Arcade-Learning-Environment
* tqdm

You'll also need to grab any roms you need.


## TODO list:

Technical improvements:
* Implement a better approximate KNN algorithm
* Add support for other environments (and alternative models)
* Add saving and loading capabilities to model+dictionary (this might include partially implementing the DND in tensorflow)

Experiments:
* Devise a way to combine the DND with a DQN where the DQN takes over in the long term
* Test with optimistically weighted value estimates as in [Particle Value Functions](https://arxiv.org/abs/1703.05820)
* Test with a count-based exploration module



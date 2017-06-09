# Neural Episodic Control

This is my attempt at replicating DeepMind's [Neural Episodic Control](https://arxiv.org/abs/1703.01988) agent. It is currently set up for running with the ALE, but can easily be adapted for other environments (you may want to use my older implementation [here](https://github.com/EndingCredits/nn_q_learning_tensorflow/blob/master/NEC.py) as a reference).

To run the code (after installing all dependencies:
```bash
python main.py --rom [path/to/rom/file.bin]
```

Further options can be found using:
```bash
python main.py -h
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
* scikit-learn (can be commented out)
* annoy
* tensorflow >1.0
* OpenCV2 (only used in the preprocessors, could be replaced with a different library)
* OpenAI Gym (if using gym)
* https://github.com/mgbellemare/Arcade-Learning-Environment (if using ALE, you'll also need to grab any roms you need.)
* tqdm


## TODO list:

Technical improvements:
* <s>Implement a better approximate KNN algorithm</s>
  * Done!
* <s>Add support for other environments (and alternative models)</s>
  * <s>In progress, almost done!</s>
  * Done!
* <s>Merge history handling with saved trajectories and replay memory to save memory</s>
  * Done!
* Replace saved trajectories (as list) with a trajectories class which also handles computing returns.
* Add saving and loading capabilities to model+dictionary (this might include partially implementing the DND in tensorflow)

Experiments:
* Decay old elements in the dictionary to simulate alpha-updates
  * Implemented and tested basic version
* Devise a way to combine the DND with a DQN where the DQN takes over in the long term
* Test with optimistically weighted value estimates as in [Particle Value Functions](https://arxiv.org/abs/1703.05820)
* Test with a count-based exploration module



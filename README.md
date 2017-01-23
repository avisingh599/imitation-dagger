##Imitation Learning with Dataset Aggregation (DAGGER)

This repository implements a simple algorithm for imitation learning: [DAGGER](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)

The implementation is tested on the problem of controling the steering wheel of a vehicle (in simulation) from pixels alone. TORCS is used as the driving simulator, and Keras is used for training a Convolutional Neural Network on the steering anlge prediction task.

##Usage
`python dagger.py`

##Requirements

1. Python 3.0
2. Keras with Tensorflow backend
3. Numpy
4. GymTORCS

##Results

After Iteration 1, crashes after 078 steps

![](http://i.imgur.com/YfqFXQZ.gif)

After Iteration 2, crashes after 151 steps

![](http://i.imgur.com/0bXKyVx.gif)

After Iteration 3, crashes after 395 steps

![](http://i.imgur.com/doz8U0z.gif)

After Iteration 4, 999 steps (complete lap, no crashing)

![](http://i.imgur.com/pKeVxLY.gif)

Note: The images fed to the ConvNet are 64x64, the images shown above have been resized to 256x256 for viewing purposes.
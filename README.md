## Imitation Learning with Dataset Aggregation (DAGGER)

This repository implements a simple algorithm for imitation learning: [DAGGER](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)

The implementation is tested on the problem of controlling the steering wheel of a vehicle (in simulation) from pixels alone. TORCS is used as the driving simulator, and Keras is used for training a Convolutional Neural Network on the steering angle prediction task.

## Usage
`python dagger.py`

## Requirements

1. Python 3.0
2. Keras with Tensorflow backend
3. Numpy
4. [Gym-TORCS](https://github.com/ugo-nama-kun/gym_torcs)

## Results

After Iteration 1, crashes after 78 steps

![](http://i.imgur.com/YfqFXQZ.gif)

After Iteration 2, crashes after 151 steps

![](http://i.imgur.com/0bXKyVx.gif)

After Iteration 3, crashes after 395 steps

![](http://i.imgur.com/doz8U0z.gif)

After Iteration 4, the car does not crash anymore: [gif](http://i.imgur.com/pKeVxLY.gif). Image cannot be displayed on the README since it exceeds the content length allowed by Github. 

Note: The images fed to the ConvNet are 64x64, the images shown above have been resized to 256x256 for viewing purposes.

from gym_torcs import TorcsEnv
import numpy as np

def get_teacher_action(ob):
    steer = ob.angle*10/np.pi
    steer -= ob.trackPos*0.10
    return np.array([steer])

img_dim = [64,64,3]
action_dim = 1
images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
actions_all = np.zeros((0,action_dim))
rewards_all = np.zeros((0,))

img_list = []
action_list = []
reward_list = []
steps = 1000

env = TorcsEnv(vision=True, throttle=False)
ob = env.reset(relaunch=True)

print('Collecting data...')
for i in range(steps):
    if i == 0:
        act = np.array([0.0])
    else:
        act = get_teacher_action(ob)

    if i%100 == 0:
        print(i)
    ob, reward, done, _ = env.step(act)
    img_list.append(ob.img)
    action_list.append(act)
    reward_list.append(np.array([reward]))

env.end()

print('Packing data into arrays...')
for img, act, rew in zip(img_list, action_list, reward_list):
    _img = np.transpose(img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, 64, 64, 3))
    images_all = np.concatenate([images_all, _img], axis=0)
    actions_all = np.concatenate([actions_all, np.reshape(act, [1,action_dim])], axis=0)
    rewards_all = np.concatenate([rewards_all, rew], axis=0)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

batch_size = 32
nb_epoch = 100

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=img_dim))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(action_dim))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])

model.fit(images_all, actions_all,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              #validation_data=(X_test, Y_test),
              shuffle=True)

#evaluate

env = TorcsEnv(vision=True, throttle=False)
ob = env.reset(relaunch=True)
reward_sum = 0.0
for i in range(steps):
    _img = np.transpose(ob.img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, 64, 64, 3))
    act = model.predict(_img)
    ob, reward, done, _ = env.step(act)
    if done is True:
        break
    reward_sum += reward
    print(i, reward, reward_sum, done, str(act[0]))

env.end()
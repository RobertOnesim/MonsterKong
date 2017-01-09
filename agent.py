import argparse
import json
import random
import numpy as np
from collections import deque

import argparse
import skimage as skimage
from skimage import transform, color, exposure

from keras.initializations import normal
from keras.layers import Convolution2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam

from  ple.games.monsterkong import MonsterKong
from ple import PLE

ACTIONS = 5 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0 # final value of epsilon
INITIAL_EPSILON = 1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                            border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init=lambda shape, name: normal(shape, scale=0.01, name=name),
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(5, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    adam = Adam(lr=1e-6)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model


ladderMaxDist = 180.0
ladderValue = 2.5

playerStartY = 441.0
levelHeight = 75.0
levelFullValue = 7.5
levelClimbValue = levelFullValue - ladderValue

coinMaxDist = 80.0
coinWeightY = 5.0
coinValue = 3.0

def getDetailedScore(game, screen):
    extraPoints = 0.0
    playerPos = game.Players[0].getPosition()

    # reward superior levels
    deltaY = playerStartY - playerPos[1]
    if game.Players[0].isJumping and not game.Players[0].onLadder:
        deltaY -= deltaY % levelHeight
    extraPoints += (deltaY // levelHeight) * levelFullValue
    extraPoints += (deltaY % levelHeight) / levelHeight * levelClimbValue

    # reward ladder proximity
    bestReward = 0.0
    if game.Players[0].onLadder:
        bestReward = ladderValue
    else:
        bestPos = game.Ladders[0].getPosition()
        for ladder in game.Ladders:
            pos = ladder.getPosition()
            if pos[1] >= playerPos[1] - 1:
                if pos[1] <= bestPos[1] and abs(playerPos[0] - pos[0]) < abs(playerPos[0] - bestPos[0]):
                    bestPos = pos
        bestReward = (ladderMaxDist - abs(playerPos[0] - bestPos[0])) / ladderMaxDist * ladderValue
        if bestReward < 0:
            bestReward = 0
    extraPoints += bestReward

    # reward coin proximity
    bestReward = 0.0
    for coin in game.Coins:
        pos = coin.getPosition()
        dist = abs(pos[0] - playerPos[0]) + abs(pos[1] - playerPos[1]) * coinWeightY
        reward = (coinMaxDist - dist) / coinMaxDist * coinValue
        if reward > bestReward:
            bestReward = reward
    extraPoints += bestReward

    return extraPoints

def trainNetwork(model,args):
    # open up a game state to communicate with ple env
    game = MonsterKong()
    p = PLE(game, fps=30, display_screen=True)

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t = p.getScreenRGB()
    # x_t, r_0, terminal = game.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    actions = p.getActionSet()
    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 1
        r_t = 0
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon or t < OBSERVE:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward

        #x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        r_t = p.act(actions[action_index])
        terminal = p.game_over()
        x_t1_colored = p.getScreenRGB()
        r_t += getDetailedScore(game.newGame, x_t1_colored)

        if terminal:
            r_t = -1000
            p.reset_game()

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 100 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()
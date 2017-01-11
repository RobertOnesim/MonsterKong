import sys
import json
import random
import numpy as np
from collections import deque

import skimage as skimage
from skimage import transform, color, exposure

from keras.initializations import normal
from keras.layers import Convolution2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam

from ple.games.monsterkong import MonsterKong
from ple import PLE


class MonsterKongPlayer:

    ACTIONS = 5             # number of valid actions
    GAMMA = 0.99            # decay rate of past observations
    OBSERVATION = 3200.     # timesteps to observe before training
    EXPLORE = 3000000.      # frames over which to anneal epsilon
    EPSILON_INITIAL = 0.2   # starting value of epsilon
    EPSILON_FINAL = 0.0001  # final value of epsilon
    SECOND_BEST = 0.25      # chance to pick the second best move
    REPLAY_MEMORY = 20000   # number of previous transitions to remember
    BATCH = 20              # size of minibatch

    PIXELCOUNT_X = 80       # resize width
    PIXELCOUNT_Y = 80       # resize height

    LADDER_DIST_MAX = 180.  # the maximum distance that will award points for ladder proximity
    LADDER_VALUE = 2.5      # the maximum value of ladder proximity

    PLAYER_START_Y = 441.   # player starting position on y axis
    LEVEL_HEIGHT = 75.      # pixel distance between levels
    LEVEL_VALUE_FULL = 7.5  # value awarded for each level
    LEVEL_VALUE_CLIMB = LEVEL_VALUE_FULL - LADDER_VALUE # partial value awarded while climbing

    COIN_DIST_MAX = 80.     # the maximum distance that will award points for coin proximity
    COIN_WEIGHT_Y = 5.      # the weight of the y axis in distance calculation
    COIN_VALUE = 3.0        # the maximum value of coin proximity

    def __init__(self):
        self.buildModel()
        self.game = MonsterKong()
        self.p = PLE(self.game, fps=30, display_screen=True)


    def buildModel(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same', input_shape=(4, self.PIXELCOUNT_X, self.PIXELCOUNT_Y)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(5, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

        self.model.compile(loss='mse', optimizer=Adam(lr=1e-6))


    def startNetwork(self, observe, epsilon):

        replayDeque = deque()

        self.p.act(self.p.NOOP)
        imageColored = self.p.getScreenRGB()

        imageNormalised = skimage.color.rgb2gray(imageColored)
        imageNormalised = skimage.transform.resize(imageNormalised, (self.PIXELCOUNT_X, self.PIXELCOUNT_Y))
        imageNormalised = skimage.exposure.rescale_intensity(imageNormalised, out_range=(0,255))

        imageList = np.stack((imageNormalised, imageNormalised, imageNormalised, imageNormalised), axis=0)

        actions = self.p.getActionSet()
        imageList = imageList.reshape(1, imageList.shape[0], imageList.shape[1], imageList.shape[2])

        frameIndex = 0
        while (True):

            #choose an action epsilon greedy
            if random.random() <= epsilon or frameIndex <= observe:
                actionIndex = random.randrange(self.ACTIONS)
            else:
                prediction = self.model.predict(imageList)
                actionIndex = np.argmax(prediction)
                # pick the second best move
                if random.random() <= self.SECOND_BEST:
                    prediction[actionIndex] = 0
                    actionIndex = np.argmax(prediction)

            #We reduced the epsilon gradually
            if epsilon > self.EPSILON_FINAL and frameIndex > observe:
                epsilon -= 1/self.EXPLORE
            #epsilon -= (EPSILON_INITIAL - EPSILON_FINAL) / EXPLORE

            #run the selected action and observed next state and reward
            actionScore = self.p.act(actions[actionIndex])
            imageColored = self.p.getScreenRGB()
            actionScore += self.getDetailedScore(imageColored)

            terminal = self.p.game_over()
            if terminal:
                actionScore = -1000
                self.p.reset_game()

            imageNormalised = skimage.color.rgb2gray(imageColored)
            imageNormalised = skimage.transform.resize(imageNormalised,(80,80))
            imageNormalised = skimage.exposure.rescale_intensity(imageNormalised, out_range=(0, 255))

            imageNormalised = imageNormalised.reshape(1, 1, imageNormalised.shape[0], imageNormalised.shape[1])
            imageListNext = np.append(imageNormalised, imageList[:, :3, :, :], axis=1)

            # store the transition in replayDeque
            replayDeque.append((imageList, actionIndex, actionScore, imageListNext, terminal))
            if len(replayDeque) > self.REPLAY_MEMORY:
                replayDeque.popleft()

            #only train if done observing
            if frameIndex > observe:
                #sample a minibatch to train on
                minibatch = random.sample(replayDeque, self.BATCH)

                inputs = np.zeros((self.BATCH, imageList.shape[1], imageList.shape[2], imageList.shape[3]))
                targets = np.zeros((inputs.shape[0], self.ACTIONS))

                #Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal = minibatch[i][4]
                    # if terminated, only equals reward

                    inputs[i:i + 1] = state_t    #I saved down imageList

                    targets[i] = self.model.predict(state_t)  # Hitting each buttom probability
                    prediction = self.model.predict(state_t1)

                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + self.GAMMA * np.max(prediction)

                self.model.train_on_batch(inputs, targets)

            imageList = imageListNext
            frameIndex = frameIndex + 1

            print("FRAME", frameIndex, "/ EPSILON", epsilon, "/ ACTION", actionIndex, "/ REWARD", actionScore)

            # save progress
            if frameIndex % 1000 == 0:
                print("Saving Model")
                self.model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(self.model.to_json(), outfile)


    def getDetailedScore(self, screen):
        extraPoints = 0.0
        playerPos = self.game.newGame.Players[0].getPosition()
        # reward superior levels
        deltaY = self.PLAYER_START_Y - playerPos[1]
        if self.game.newGame.Players[0].isJumping and not self.game.newGame.Players[0].onLadder:
            deltaY -= deltaY % self.LEVEL_HEIGHT
        extraPoints += (deltaY // self.LEVEL_HEIGHT) * self.LEVEL_VALUE_FULL
        extraPoints += (deltaY % self.LEVEL_HEIGHT) / self.LEVEL_HEIGHT * self.LEVEL_VALUE_CLIMB

        # reward ladder proximity
        bestReward = 0.0
        if self.game.newGame.Players[0].onLadder:
            bestReward = self.LADDER_VALUE
        else:
            bestPos = self.game.newGame.Ladders[0].getPosition()
            for ladder in self.game.newGame.Ladders:
                pos = ladder.getPosition()
                if pos[1] >= playerPos[1] - 1:
                    if pos[1] <= bestPos[1] and abs(playerPos[0] - pos[0]) < abs(playerPos[0] - bestPos[0]):
                        bestPos = pos
            bestReward = (self.LADDER_DIST_MAX - abs(playerPos[0] - bestPos[0])) / self.LADDER_DIST_MAX * self.LADDER_VALUE
            if bestReward < 0:
                bestReward = 0
        extraPoints += bestReward

        # reward coin proximity
        bestReward = 0.0
        for coin in self.game.newGame.Coins:
            pos = coin.getPosition()
            dist = abs(pos[0] - playerPos[0]) + abs(pos[1] - playerPos[1]) * self.COIN_WEIGHT_Y
            reward = (self.COIN_DIST_MAX - dist) / self.COIN_DIST_MAX * self.COIN_VALUE
            if reward > bestReward:
                bestReward = reward
        extraPoints += bestReward

        return extraPoints


def main():
    player = MonsterKongPlayer()
    if sys.argv[1] == 'play':
        player.model.load_weights("model.h5")
        player.model.compile(loss='mse', optimizer=Adam(lr=1e-6))
        player.startNetwork(999999999, player.EPSILON_FINAL)

    elif sys.argv[1] == 'train':
        player.startNetwork(player.OBSERVATION, player.EPSILON_INITIAL)
    else:
        print('Please specify what would you like the player to do')

if __name__ == "__main__":
    main()
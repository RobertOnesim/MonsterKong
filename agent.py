import sys
import json
import random
import numpy as np
from collections import deque

from PIL import Image

from keras.initializations import normal
from keras.layers import Convolution2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam

from ple.games.monsterkong import MonsterKong
from ple import PLE


class MonsterKongPlayer:

    BATCH = 20              # size of miniBatch
    GAMMA = 0.99
    EXPLORE = 3200          # frames to observe before training
    TRAIN = 1000000         # frames to train and adjust epsilon
    SECOND_BEST = 0.25      # chance to pick the second best move
    REPLAY_MEMORY = 20000   # save previous transitions

    LADDER_DIST_MAX = 180.  # the maximum distance that will award points for ladder proximity
    LADDER_VALUE = 2.5      # the maximum value of ladder proximity

    PLAYER_START_Y = 441.   # player starting position on y axis
    LEVEL_HEIGHT = 75.      # pixel distance between levels
    LEVEL_VALUE_FULL = 7.5  # value awarded for each level
    LEVEL_VALUE_CLIMB = LEVEL_VALUE_FULL - LADDER_VALUE # partial value awarded while climbing

    COIN_DIST_MAX = 80.     # the maximum distance that will award points for coin proximity
    COIN_WEIGHT_Y = 5.      # the weight of the y axis in distance calculation
    COIN_VALUE = 3.0        # the maximum value of coin proximity

    ACTIONS = 5             # valid moves
    EPSILON_INITIAL = 0.2   # starting value of epsilon
    EPSILON_FINAL = 0.0001  # final value of epsilon


    def __init__(self):
        self.buildModel()
        self.game = MonsterKong()
        self.p = PLE(self.game, fps=30, display_screen=True)


    def buildModel(self):
        # build model using Keras
        self.model = Sequential()
        # input layer 4x80x80 image and 3 convolution hidden layers with activation function "ReLu" f(x) = max(0,x)
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same', input_shape=(4, 80, 80)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        # last hidden fully connected layer
        self.model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        self.model.add(Activation('relu'))
        # output layer -  1 neuron for each valid move (5)
        self.model.add(Dense(5, init=lambda shape, name: normal(shape, scale=0.01, name=name)))

        # using ADAM (Adaptive Moment Estimation)
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-6))

    def saveModel(self):
        # saving the model
        self.model.save_weights("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)

    def startNetwork(self, explore, epsilon):

        replayDeque = deque()

        # get first image and resize in a 80x80 image and group 4 of this image
        self.p.act(self.p.NOOP)
        imageColored = self.p.getScreenRGB()
        imageNormalised = np.array(Image.fromarray(imageColored, 'RGB').convert('L').resize((80, 80)).getdata()).reshape((80, 80))
        imageList = np.stack((imageNormalised, imageNormalised, imageNormalised, imageNormalised), axis=0)
        # Keras input 1x4x80x80 so we need to reshape the imageList
        imageList = imageList.reshape(1, imageList.shape[0], imageList.shape[1], imageList.shape[2])

        # get the action set for this game
        actions = self.p.getActionSet()

        frameIndex = 0
        while (frameIndex < self.TRAIN):

            #choose an action epsilon greedy
            if random.random() <= epsilon or frameIndex <= explore:
                actionIndex = random.randrange(self.ACTIONS)
            else:
                prediction = self.model.predict(imageList)
                actionIndex = np.argmax(prediction)
                # pick the second best move
                if random.random() <= self.SECOND_BEST:
                    prediction[actionIndex] = 0
                    actionIndex = np.argmax(prediction)

            # We reduced the epsilon gradually
            if epsilon > self.EPSILON_FINAL and frameIndex > explore:
                epsilon -= 1/self.TRAIN

            #run the selected action and observed next state and reward
            actionScore = self.p.act(actions[actionIndex])
            actionScore += self.getDetailedScore(imageColored)
            imageColored = self.p.getScreenRGB()

            # check if the game is over - yes (reset the game and give a negative reward)
            gameOver = self.p.game_over()
            if gameOver:
                actionScore = -1000
                self.p.reset_game()

            # resize the current state image
            imageNormalised = np.array(Image.fromarray(imageColored, 'RGB').convert('L').resize((80, 80)).getdata()).reshape((80, 80))
            imageNormalised = imageNormalised.reshape(1, 1, imageNormalised.shape[0], imageNormalised.shape[1])
            imageListNext = np.append(imageNormalised, imageList[:, :3, :, :], axis=1)

            # store the transition in replayDeque
            replayDeque.append((imageList, actionIndex, actionScore, imageListNext, gameOver))
            if len(replayDeque) > self.REPLAY_MEMORY:
                replayDeque.popleft()

            #only train if done observing
            if frameIndex > explore:
                #sample a miniBatch to train on
                minibatch = random.sample(replayDeque, self.BATCH)

                inputs = np.zeros((self.BATCH, imageList.shape[1], imageList.shape[2], imageList.shape[3]))
                targets = np.zeros((inputs.shape[0], self.ACTIONS))

                #Now we do the experience replay
                for s in range(0, len(minibatch)):
                    oldState = minibatch[s][0] # old state
                    action = minibatch[s][1]   # move index
                    reward = minibatch[s][2]   # reward
                    newState = minibatch[s][3] # new state
                    gameOver = minibatch[s][4] # flag game over

                    inputs[s:s + 1] = oldState    # saved imageList
                    targets[s] = self.model.predict(oldState)
                    newQval = self.model.predict(newState) # Q(s,a)

                    if gameOver:
                        targets[s, action] = reward
                    else:
                        targets[s, action] = reward + self.GAMMA * np.max(newQval)

                self.model.train_on_batch(inputs, targets) # this return the loss

            imageList = imageListNext
            frameIndex = frameIndex + 1

            print("FRAME", frameIndex, " EPSILON", epsilon, " ACTION", actionIndex, " REWARD", actionScore)

            # save model
            if frameIndex % 1000 == 0:
                self.saveModel()

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
        player.startNetwork(player.EXPLORE, player.EPSILON_INITIAL)
    else:
        print('Please specify what would you like the player to do')

if __name__ == "__main__":
    main()
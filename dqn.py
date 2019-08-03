import gym  # 倒立振子(cartpole)の実行環境
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from gym import wrappers
from keras import backend as K
import tensorflow as tf


def huberloss(y_true,y_pred):
    err = y_true -y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5*K.square(err)
    L1 = (K.abs(err)-0.5)
    loss = tf.where(cond,L2,L1)
    return K.mean(loss)


class QNetWork:
    def __init__(self,learning_rate=0.01,state_size=4,action_size=2,hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size,activation='relu',input_dim=state_size))
        self.model.add(Dense(hidden_size,activation='relu'))
        self.model.add(Dense(action_size,activation='linear'))
        self.optimizer = Adam(lr=learning_rate)

        self.model.compile(loss=huberloss,optimizer=self.optimizer)

    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            targets[i][action_b] = target               # 教師信号

        # shiglayさんよりアドバイスいただき、for文の外へ修正しました
        self.model.fit(inputs, targets, epochs=1, verbose=0)


class Memory:
    def __init__(self,max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self,experience):
        self.buffer.append(experience)

    def sample(self,batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),size=batch_size,replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)

class Actor:
    def get_action(self,state,episode,mainQN):

        epsilon = 0.001 + 0.9/(1.0+episode)

        if epsilon<=np.random.uniform(0,1):
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)
        else:
            action = np.random.choice([0,1])

        return action

print("aaa")


DQN_MODE = 0
LENDER_MODE=1

env = gym.make('CartPole-v0')
num_episodes = 399
max_number_of_steps = 200
goal_average_reward =195
num_consecutive_iterations = 10
total_reward_vec = np.zeros(num_consecutive_iterations)
gamma = 0.99
islearned = 0
isrender = 0

hidden_size = 16
learning_rate = 0.001
memory_size = 10000
batch_size = 32

mainQN = QNetWork(hidden_size=hidden_size,learning_rate=learning_rate)
targetQN = QNetWork(hidden_size=hidden_size,learning_rate=learning_rate)

memory = Memory(max_size=memory_size)
actor = Actor()

for episode in range(num_episodes):
    env.reset()
    state,reward,done ,_ =env.step(env.action_space.sample())
    state = np.reshape(state,[1,4])
    episode_reward = 0

    targetQN.model.set_weights(mainQN.model.get_weights())

    for t in range(max_number_of_steps+1):
        if (islearned == 1) and LENDER_MODE:  # 学習終了したらcartPoleを描画する
            env.render()
            time.sleep(0.1)
            print(state[0, 0])

        action = actor.get_action(state,episode,mainQN)
        next_state,reward,done,info =env.step(action)
        next_state= np.reshape(next_state,[1,4])

        if done:
            next_stete = np.zeros(state.shape)
            if t<195:
                reward =-1
            else:
                reward=1
        else:
            reward=0

        episode_reward += 1

        memory.add((state,action,reward,next_state))
        state = next_state

        if(memory.len()>batch_size)and not islearned:
            mainQN.replay(memory,batch_size,gamma,targetQN)
            #print("learn")

        if DQN_MODE:
            targetQN.model.set_weights(mainQN.model.get_weights())

        if done:

            total_reward_vec = np.hstack((total_reward_vec[1:],episode_reward))
            print("%d episode after %d mean%d"%(episode,t+1,total_reward_vec.mean()))
            break

    if total_reward_vec.mean( )>=goal_average_reward:
        print("success %d episode"%episode)
        islearned=1
        if isrender ==0:
            isrender=1

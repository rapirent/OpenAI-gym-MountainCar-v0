import gym
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n','--model_name', default='default_model')
parser.add_argument('-e','--train_episode', default='5000')
parser.add_argument('-l','--step_limit', default='1500')
parser.add_argument('-s','--state_num', default='15')
parser.add_argument('-lr','--learning_rate', default='0.01')
args = parser.parse_args()

env = gym.make('MountainCar-v0')
env.seed(21)

TRAIN_EPISODE = int(args.train_episode)
STEP_LIMIT = int(args.step_limit)

console_output = open('./console_log/' + args.model_name, 'w')

class Qlearning():

    def __init__(self):
        self._position_state = int(args.state_num)
        self._velocity_state = int(args.state_num)

        self._epsilon = 0.7              # how greedy
        self._gamma = 0.99
        self._learning_rate = float(args.learning_rate)

        self._action_space = [0, 1, 2]
        self._max_position = 0.6
        self._min_position = -1.2
        self._goal_position = 0.5
        self._max_velocity = 0.07
        self.qtable = np.zeros((self._position_state, self._velocity_state, 3))
        self.d1 = np.linspace(self._min_position, self._max_position, self._position_state+1)
        self.d2 = np.linspace(-self._max_velocity, self._max_velocity, self._velocity_state+1)

    def to_state(self, position, velocity):
        p_state = 0
        v_state = 0
        for i, j in enumerate(self.d1):
            # print(i, j)
            if j >= position:
                p_state = i-1
                break
        for i, j in enumerate(self.d2):
            # print(i, j)
            if j >= velocity:
                v_state = i-1
                break
        return (p_state, v_state)

    def choose_action(self, p_state, v_state, i_episode):
        # lookup self.table and choose an action
        # 2 case: greedy or random

        EPSILON_ = (i_episode*(1-self._epsilon) / STEP_LIMIT) + self._epsilon

        state_actions = self.qtable[p_state][v_state]

        if (np.random.uniform() > EPSILON_) or (all(a == 0 for a in state_actions)):
            action = np.random.choice(self._action_space)
            # print("random", action, action)
        else:
            action = max(state_actions)
            action = [i for i, j in enumerate(state_actions) if j == action][0]
            # print("greedy", action)
        return action

    def update_qtable(self, cur_p, cur_v, new_p, new_v, this_a, exp_a, reward):
        # qtable[3][7][1] += 1
        self.qtable[cur_p][cur_v][this_a] = (self.qtable[cur_p][cur_v][this_a] +
                                            self._learning_rate * (reward + self._gamma*self.qtable[new_p][new_v][exp_a] - self.qtable[cur_p][cur_v][this_a]))

model = Qlearning()

total_reward_list = []
total_step_list = []
fail = []

for i in range(TRAIN_EPISODE):
    total_reward = 0
    total_step = 0

    observation = env.reset()

    cur_position = observation[0]
    cur_velocity = observation[1]
    cur_p, cur_v = model.to_state(cur_position, cur_velocity)
    a = model.choose_action(cur_p, cur_v, i)
    for j in range(STEP_LIMIT):

        observation_after, reward, done, info = env.step(a)
        total_step += 1
        new_position = observation_after[0]
        new_velocity = observation_after[1]
        new_p, new_v = model.to_state(new_position, new_velocity)

        total_reward += reward
        # use new state to choose a expect action:
        a_expect = model.choose_action(new_p, new_v, i)

        # update qtable
        model.update_qtable(cur_p, cur_v, new_p, new_v, a, a_expect, reward)
        cur_p = new_p
        cur_v = new_v
        a = a_expect
        # env.render()_
        if new_position >= 0.5:
            total_step_list.append(total_step)
            total_reward_list.append(total_reward)
            fail.append(0)
            break
    if new_position < 0.5:
        total_step_list.append(float('nan'))
        total_reward_list.append(float('nan'))
        fail.append(1)
    print(i, "th train finish in reward: ", total_reward, 'steps: ', total_step, file=console_output)

plt.title('reward')
plt.ylabel('reward')
plt.xlabel('epoch')
plt.plot(total_reward_list)
plt.savefig('./fig/' + args.model_name + '-reward.png')
plt.clf()

plt.title('total steps')
plt.ylabel('steps')
plt.xlabel('epoch')
plt.plot(total_step_list)
plt.savefig('./fig/' + args.model_name + '-step.png')
plt.clf()

plt.title('total fail times')
plt.ylabel('fail')
plt.xlabel('epoch')
plt.plot(fail)
plt.savefig('./fig/' + args.model_name + '-fail.png')
plt.clf()

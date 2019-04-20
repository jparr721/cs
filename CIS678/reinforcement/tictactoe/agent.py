#!/usr/bin/env python

import gym
import numpy as np
from gym.envs.registration import register

register(
        id='tic-tac-toe-v0',
        entry_point='envs:TicTacToeEnv'
        )
env = gym.make('tic-tac-toe-v0')


def q(eps=1000, gamma=0.9, eta=0.8, max_t=1000):
    env.reset()

    Q = np.zeros(shape=(3, 3))

    rewards = []

    for i in range(eps):
        print(f'Episode {i} out of {eps} total')
        state = env.reset()
        r_all = 0
        for j in range(max_t):
            action = np.argmax(
                    Q[state, :] + np.random.randn(1, 2) * (1./(i + 1)))

            next_state, reward, done, _ = env.step(action)
            Q[state, action] = Q[state, action] + \
                eta * (
                    reward + gamma * np.max(
                        Q[next_state, :] - Q[state, action]))
            r_all += reward
            state = next_state
            if done:
                break
        rewards.append(r_all)
    env.render()

    return Q


if __name__ == '__main__':
    q()

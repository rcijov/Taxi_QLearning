import gym
import numpy as np
from collections import defaultdict
import random

env = gym.make('Taxi-v2')

init_state = env.reset()
env.render()

alpha = 0.1
gamma = 0.6
epsilon = 0.8


q_table = np.zeros([env.observation_space.n, env.action_space.n])

def select_optimal_action(q_table, state):

    if np.sum(q_table[state]) == 0:
        return random.randint(0, q_table.shape[1] - 1)
    
    return np.argmax(q_table[state])


def q_learning_update(q_table, env, state, epsilon):

    if random.uniform(0, 1) > epsilon:
        action = env.action_space.sample()
    else:
        action = select_optimal_action(q_table, state)
    
    next_state, reward, done, _ = env.step(action)
    old_q_value = q_table[state][action]

    next_max = np.max(q_table[next_state])

    new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)

    q_table[state][action] = new_q_value

    return next_state, reward, done


def train_agent(q_table, env, num_episodes, epsilon):
    
    for i in range(num_episodes):
        state = env.reset()

        epochs = 0
        num_penalties, total_reward = 0, 0

        done = False

        while not done:
            state, reward, done = q_learning_update(q_table, env, state, epsilon)
            total_reward += reward

            if reward == -10:
                num_penalties += 1
            
            epochs += 1
        
        print("\nTraining episode {}",format(i + 1))
        print("Time steps: {}, Penalties: {}, Reward: {}".format(epochs,
                                                            num_penalties,
                                                            total_reward))
    print("Training finished.\n")
    return q_table

q_table = train_agent(q_table, env, 5000, epsilon)
import gym
import numpy as np
import random
from random import uniform
from collections import defaultdict

env = gym.make('Taxi-v2')

init_state = env.reset()

print("Decoded state", list(env.env.decode(init_state)))

env.render()

alpha = 0.1
gamma = 0.6
epsilon = 0.8

q_table = np.zeros([env.observation_space.n, env.action_space.n])

def q_learning_update(q_table, env, state, epsilon):
    """
    Updates the Q-values according to the Q-learning equation.
    """
    if uniform(0, 1) > epsilon:
        action = env.action_space.sample() # select a random action
    else:
        action = select_optimal_action(q_table, state) # select an optimal action
    
    next_state, reward, done, _ = env.step(action)
    old_q_value = q_table[state][action]


    # Maximum q_value for the actions in next state
    next_max = np.max(q_table[next_state])

    new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)

    # Update the q_value
    q_table[state][action] = new_q_value

    return next_state, reward, done


def select_optimal_action(q_table, state):
    """
    Given a state, select the action from the action space having the
    highest Q-value in the q_table.
    """
    if np.sum(q_table[state]) == 0:
        return random.randint(0, q_table.shape[1]-1)
    
    return np.argmax(q_table[state])



def train_agent(q_table, env, num_episodes, epsilon):
    for i in range(num_episodes):
        state = env.reset()

        done = False
        while not done:
            state, reward, done = q_learning_update(q_table, env, state, epsilon)
    
    return q_table



def run_trials(q_table, env, num_trials):
    data_by_episode = []

    for i in range(num_trials):
        state = env.reset()
        epochs, num_penalties, episode_reward = 0, 0, 0
        
        done = False
        while not done:
            next_action = select_optimal_action(q_table, state)
            state, reward, done, _ = env.step(next_action)

            if reward == -10:
                num_penalties += 1

            epochs += 1
            episode_reward += reward
        
        data_by_episode.append((epochs, num_penalties, episode_reward))
    
    return data_by_episode


def calculate_stats(trial_data):
    
    epochs, penalties, rewards = zip(*trial_data)
    total_epochs, total_penalties, total_reward = sum(epochs), sum(penalties), sum(rewards)
    num_trials = len(epochs)
    average_time = total_epochs / float(num_trials)
    average_penalties = total_penalties / float(num_trials)
    average_reward_per_move = total_reward/float(total_epochs)
    
    return (average_time, average_penalties, average_reward_per_move)


data_to_plot = []
train_episodes_before_eval = 100
num_episodes_for_eval = 50

while True:

    q_table = train_agent(q_table, env, train_episodes_before_eval, epsilon=epsilon)
    trial_data = run_trials(q_table, env, num_episodes_for_eval)
    
    #env.render()

    average_time, average_penalties, average_reward_per_move = calculate_stats(trial_data)
    
    data_to_plot.append((average_time, average_penalties, average_reward_per_move))
    
    if average_penalties == 0 and average_reward_per_move > 0.8:
        print("Average time steps taken: {}".format(average_time))
        print("Average number of penalties incurred: {}".format(average_penalties))
        print("Average reward per move: {}".format(average_reward_per_move))
        break


with open("q_table.save", "wb") as f:
    np.save(f, q_table)
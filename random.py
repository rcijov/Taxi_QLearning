import gym
import time

env = gym.make('Taxi-v2')

env.reset()
env.render()

epochs, num_penalties, total_reward = 0,0,0
episodes = 100
curr_episode = 0

while curr_episode < episodes:
    reward = 0
    done = False

    while not done:
        action = env.action_space.sample()

        state, reward, done, info = env.step(action)

        if reward == -10:
            num_penalties += 1
        
        epochs += 1
        total_reward += reward
        env.render()
    
    curr_episode += 1
    env.reset()

print("Total rewards ", total_reward, " epochs ", epochs)
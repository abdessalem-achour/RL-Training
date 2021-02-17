import gym
import numpy as np

# existing environment
env=gym.make("MountainCar-v0") 

# initilaze environment (mandatory)
env.reset() 

#environment informations
print(env.action_space)
print(env.action_space.n)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

# Space descretization
DISCRETE_OS_SIZE = [20]*len(env.observation_space.high) # definite number * number of space observations
discret_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print("discret_os_win_size = ", discret_os_win_size)

# Q-table intialization
q_table = np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print("q_table shape = ", q_table.shape)
print(q_table)


# target not reached yet
'''done = False
t = 0

while not done:
    action = 2
    new_state,reward,done,_= env.step(action)
    print(new_state)
    env.render()
    t += 1
    if done:
        print("Episode finished after {} timesteps".format(t))
        break'''

env.close()
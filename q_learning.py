import gym
import numpy as np

# existing environment
env = gym.make("MountainCar-v0") 

# initilaze environment (mandatory)
state = env.reset() 

# Q_fuction parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95 
EPISODES = 25000
SHOW_EVERY = 2000

# Other parameters
epsilon = 0.5 # exploration rate
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# environment informations
'''print(env.action_space)
print(env.action_space.n)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)'''

# Space descretization
DISCRETE_OS_SIZE = [20]*len(env.observation_space.high) # definite number * number of space observations
discret_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print("discret_os_win_size = ", discret_os_win_size)

def get_descret_state(state):
    descrete_state = (state - env.observation_space.low)/ discret_os_win_size
    return tuple(descrete_state.astype(int))

# Q-table intialization
q_table = np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n]))
#print("q_table shape = ", q_table.shape)
#print(q_table)
#print(q_table[descrete_state])


# Q_learning algorithm
for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else :
        render = False

    descrete_state = get_descret_state(env.reset())
    #print("descrete state = ",descrete_state)
    done = False
    while not done:

        if np.random.random()> epsilon :
            action = np.argmax(q_table[descrete_state]) # action with biggest Q value
        else :
            action = np.random.randint(0,env.action_space.n)

        new_state,reward,done,_= env.step(action)
        new_descrete_state = get_descret_state(new_state)
        if render :
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_descrete_state])
            current_q = q_table[descrete_state +(action,)]
            new_q = (1- LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q )
            q_table[descrete_state+(action,)] = new_q
        elif new_state[0] >= env.goal_position: # target reached
            q_table[descrete_state+(action,)] = 0
            print(f"Gaol reached in espisode {episode}.")
            
        descrete_state = new_descrete_state     # update the descrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING :
        epsilon -= epsilon_decay_value
        
env.close()
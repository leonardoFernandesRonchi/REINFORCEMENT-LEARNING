import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)


q_table = np.zeros((
    env.observation_space.n,
    env.action_space.n
))


alpha = 0.1
gamma = 0.99
epsilon = 0.1


for episode in range (100):
    state, _ = env.reset()
        
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        
        done = terminated or truncated
        
        best_next_action = np.argmax(q_table[next_state])
        
        
        td_target = (
            reward
            + gamma * q_table[next_state, best_next_action]
        )
        
        print(td_target)
        
        q_table[state, action] += (
            alpha
            * (td_target - q_table[state, action])
        )
        
        state = next_state
        
        
env.close            
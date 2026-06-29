import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n

q_table = np.zeros((n_states, n_actions))

print("Nombre d'états  :", n_states)
print("Nombre d'actions:", n_actions)
print("Forme Q-table   :", q_table.shape)
print("Q-table initiale :")
print(q_table)

env.close()
import gymnasium as gym

env = gym.make("CartPole-v1")

print(">>> Espace d'observation")
print("Type :", type(env.observation_space).__name__)
print("Détail :", env.observation_space)
print("Borne basse  :", env.observation_space.low)
print("Borne haute  :", env.observation_space.high)
print("Forme (shape):", env.observation_space.shape)
print("Exemple d'observation aléatoire :", env.observation_space.sample())

print()

print(">>> Espace d'action")
print("Type :", type(env.action_space).__name__)
print("Détail :", env.action_space)
print("Nombre d'actions possibles :", env.action_space.n)
print("Exemple d'action aléatoire :", env.action_space.sample())

env.close()
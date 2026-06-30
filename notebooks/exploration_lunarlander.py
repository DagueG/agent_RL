import gymnasium as gym

env = gym.make("LunarLander-v3")

print(">>> Espace d'observation")
print("Type :", type(env.observation_space).__name__)
print("Détail :", env.observation_space)
print("Forme :", env.observation_space.shape)
print("Borne basse :", env.observation_space.low)
print("Borne haute :", env.observation_space.high)
print("Exemple :", env.observation_space.sample())

print("\n>>> Espace d'action")
print("Type :", type(env.action_space).__name__)
print("Détail :", env.action_space)
print("Nombre d'actions :", env.action_space.n)

print("\n>>> Test d'un épisode aléatoire")
state, _ = env.reset(seed=42)
print("État initial :", state)

total_reward = 0.0
steps = 0
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    steps += 1

print(f"Épisode terminé après {steps} étapes")
print(f"Récompense totale : {total_reward:.2f}")
print(f"État final : {state}")

env.close()
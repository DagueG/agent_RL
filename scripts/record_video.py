import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

os.makedirs("videos", exist_ok=True)

# Wrapper RecordVideo : enregistre la vidéo de chaque épisode
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="videos",
    name_prefix="eagle1_landing",
    episode_trigger=lambda ep_id: True,  # enregistre tous les épisodes
)

model = DQN.load("models/best_model.zip")

# On joue plusieurs épisodes pour en avoir un de qualité (20-30s = ~400-600 steps à 50fps)
N_EPISODES = 5
results = []

for ep in range(N_EPISODES):
    obs, _ = env.reset(seed=42 + ep)
    terminated = truncated = False
    total_reward = 0.0
    steps = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

    results.append((ep, total_reward, steps))
    print(f"Épisode {ep} | reward {total_reward:7.2f} | steps {steps}")

env.close()

print("\nVidéos disponibles dans le dossier videos/")
print("\nSuggestion : garde l'épisode avec la meilleure récompense et ~300-500 steps")
print("(à 50 fps cela fait 6-10s ; multiplier les épisodes si besoin pour atteindre 20-30s).")
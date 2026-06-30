import csv
import os

import gymnasium as gym
from stable_baselines3 import DQN

os.makedirs("data", exist_ok=True)

env = gym.make("LunarLander-v3")
model = DQN.load("models/best_model.zip")

N_EPISODES = 100
rows = []

for ep in range(1, N_EPISODES + 1):
    obs, _ = env.reset(seed=ep)
    terminated = truncated = False
    total_reward = 0.0
    steps = 0
    last_obs = obs

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        last_obs = obs

    # Critères : récompense >= 200 = atterrissage réussi (cf brief)
    success = total_reward >= 200
    # Atterrissage "doux" = les deux jambes en contact (indices 6 et 7 dans l'obs)
    soft_landing = bool(last_obs[6]) and bool(last_obs[7])

    rows.append({
        "episode": ep,
        "reward": round(total_reward, 2),
        "steps": steps,
        "success": int(success),
        "soft_landing": int(soft_landing),
        "final_x": round(float(last_obs[0]), 3),
        "final_y": round(float(last_obs[1]), 3),
    })

    if ep % 10 == 0:
        print(f"Épisode {ep:3d} | reward {total_reward:7.2f} | steps {steps:3d}")

env.close()

# Sauvegarde CSV
csv_path = "data/eval_logs.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Stats globales
import statistics
rewards = [r["reward"] for r in rows]
print(f"\nFichier sauvegardé : {csv_path}")
print(f"Récompense moyenne : {statistics.mean(rewards):.2f}")
print(f"Écart-type         : {statistics.stdev(rewards):.2f}")
print(f"Taux de succès     : {sum(r['success'] for r in rows)}/{N_EPISODES}")
print(f"Atterrissages doux : {sum(r['soft_landing'] for r in rows)}/{N_EPISODES}")
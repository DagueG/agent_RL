import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)
q_table = np.load("q_table.npy")

n_eval_episodes = 100
max_steps = 100
total_wins = 0

for episode in range(n_eval_episodes):
    state, _ = env.reset()
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = np.argmax(q_table[state, :])   # pure exploitation
        state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            if reward == 1.0:
                total_wins += 1
            break

env.close()

success_rate = total_wins / n_eval_episodes * 100
print(f"Taux de réussite sur {n_eval_episodes} épisodes : {success_rate:.0f}%")
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))

# Hyperparamètres
learning_rate = 0.1
discount_factor = 0.99
n_episodes = 10000
max_steps = 100

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

rewards_per_episode = []

for episode in range(n_episodes):
    state, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    for _ in range(max_steps):
        # Stratégie epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()           # exploration
        else:
            action = np.argmax(q_table[state, :])        # exploitation

        new_state, reward, terminated, truncated, _ = env.step(action)

        # Mise à jour Bellman
        old_value = q_table[state, action]
        future_max = np.max(q_table[new_state, :])
        q_table[state, action] = old_value + learning_rate * (
            reward + discount_factor * future_max - old_value
        )

        state = new_state
        total_reward += reward

        if terminated or truncated:
            break

    rewards_per_episode.append(total_reward)

    # Décroissance d'epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

env.close()

# Sauvegarde de la Q-table pour l'étape 3
np.save("q_table.npy", q_table)

# Affichage de la progression par tranches de 1000 épisodes
print("Taux de réussite moyen par tranche de 1000 épisodes :")
rewards_per_episode = np.array(rewards_per_episode)
for i in range(0, n_episodes, 1000):
    print(f"  épisodes {i:5d}-{i+999:5d} : {rewards_per_episode[i:i+1000].mean():.3f}")

print(f"\nEpsilon final : {epsilon:.4f}")
print("\nQ-table apprise :")
print(np.round(q_table, 3))
import gymnasium as gym

env = gym.make("CartPole-v1")

n_episodes = 10

for episode in range(1, n_episodes + 1):
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Épisode {episode:2d} | étapes : {steps:3d} | récompense totale : {total_reward}")

env.close()
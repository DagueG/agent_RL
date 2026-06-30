import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.monitor import Monitor


# 1) Environnement
env = gym.make("CartPole-v1")

# 2) Instanciation du modèle DQN
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=2.3e-3,
    batch_size=64,
    buffer_size=100000,
    learning_starts=1000,
    gamma=0.99,
    target_update_interval=10,
    train_freq=256,
    gradient_steps=128,
    exploration_fraction=0.16,
    exploration_final_eps=0.04,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log="./logs/",
)

# 3) Entraînement
model.learn(total_timesteps=50000)

# 4) Évaluation sur 100 épisodes
eval_env = Monitor(gym.make("CartPole-v1"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"\nRécompense moyenne sur 100 épisodes : {mean_reward:.2f} +/- {std_reward:.2f}")

# 5) Sauvegarde
model.save("dqn_cartpole")
print("Modèle sauvegardé dans dqn_cartpole.zip")

env.close()
eval_env.close()
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = gym.make("LunarLander-v3")

model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./logs/baseline_dqn/",
)

model.learn(total_timesteps=50000)

eval_env = Monitor(gym.make("LunarLander-v3"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
print(f"\n[BASELINE] Récompense moyenne sur 50 épisodes : {mean_reward:.2f} +/- {std_reward:.2f}")

model.save("models/baseline_dqn")
print("Modèle sauvegardé dans models/baseline_dqn.zip")

env.close()
eval_env.close()
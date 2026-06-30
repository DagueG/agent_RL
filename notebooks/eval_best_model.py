import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

eval_env = Monitor(gym.make("LunarLander-v3"))

for name in ["models/best_model.zip", "models/tuned_dqn_final.zip"]:
    model = DQN.load(name)
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
    print(f"{name:40s} : {mean:7.2f} +/- {std:6.2f}")

eval_env.close()
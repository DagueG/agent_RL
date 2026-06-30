import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("LunarLander-v3"))

# Hyperparamètres RL Zoo pour LunarLander
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=6.3e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    gamma=0.99,
    target_update_interval=250,
    train_freq=4,
    gradient_steps=-1,
    exploration_fraction=0.12,
    exploration_final_eps=0.10,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log="./logs/tuned_dqn/",
)

# Callback : évalue tous les 10000 pas et sauvegarde le meilleur modèle
eval_env = Monitor(gym.make("LunarLander-v3"))
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/tuned_dqn/eval/",
    eval_freq=10000,
    n_eval_episodes=20,
    deterministic=True,
    render=False,
)

model.learn(total_timesteps=200000, callback=eval_callback)

# Évaluation finale sur 100 épisodes (critère du brief)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"\n[TUNED] Récompense moyenne sur 100 épisodes : {mean_reward:.2f} +/- {std_reward:.2f}")

model.save("models/tuned_dqn_final")
print("Modèle sauvegardé dans models/tuned_dqn_final.zip")
print("Meilleur modèle (selon EvalCallback) dans models/best_model.zip")

env.close()
eval_env.close()
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from exercice3_etape1 import DQN, ReplayBuffer


# ---------- Hyperparamètres ----------
N_EPISODES = 500
GAMMA = 0.99                # facteur de discount
LR = 1e-3                   # learning rate
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
MIN_BUFFER_BEFORE_TRAIN = 1000
TARGET_UPDATE_FREQ = 10     # tous les N épisodes, on copie policy_net -> target_net

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")


# ---------- Initialisation ----------
env = gym.make("CartPole-v1")
n_obs = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(n_obs, n_actions).to(device)   # réseau qui agit et apprend
target_net = DQN(n_obs, n_actions).to(device)   # réseau cible (poids figés sauf sync)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
buffer = ReplayBuffer(BUFFER_CAPACITY)

epsilon = EPS_START
episode_rewards = []


# ---------- Sélection d'action (epsilon-greedy) ----------
def select_action(state, eps):
    if random.random() < eps:
        return env.action_space.sample()
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_t)
        return int(q_values.argmax(dim=1).item())


# ---------- Étape d'optimisation ----------
def optimize_model():
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return

    # 1) Échantillonner un batch
    batch = buffer.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states      = torch.FloatTensor(np.array(states)).to(device)
    actions     = torch.LongTensor(actions).unsqueeze(1).to(device)        # (B, 1)
    rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(device)       # (B, 1)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones       = torch.FloatTensor(dones).unsqueeze(1).to(device)         # (B, 1)

    # 2) Q(s, a) prédit par policy_net pour les actions effectivement prises
    q_values = policy_net(states).gather(1, actions)                       # (B, 1)

    # 3) Cible : r + gamma * max_a' Q_target(s', a'), nulle si done
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1, keepdim=True)[0]       # (B, 1)
        target = rewards + GAMMA * next_q * (1.0 - dones)

    # 4) Perte + descente de gradient
    loss = loss_fn(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---------- Boucle d'entraînement ----------
for episode in range(1, N_EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward

        optimize_model()

    episode_rewards.append(total_reward)
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    # Synchronisation périodique du target_net
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 50 == 0:
        avg = np.mean(episode_rewards[-50:])
        print(f"Épisode {episode:4d} | récompense moyenne (50 derniers) : {avg:6.1f} | epsilon : {epsilon:.3f}")

env.close()

# ---------- Sauvegarde du modèle ----------
torch.save(policy_net.state_dict(), "dqn_manuel_cartpole.pth")
print("\nModèle sauvegardé dans dqn_manuel_cartpole.pth")

# ---------- Courbe des récompenses ----------
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.4, label="Récompense par épisode")
window = 20
if len(episode_rewards) >= window:
    moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    plt.plot(range(window - 1, len(episode_rewards)), moving_avg, label=f"Moyenne mobile ({window})")
plt.xlabel("Épisode")
plt.ylabel("Récompense totale")
plt.title("DQN manuel sur CartPole-v1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("dqn_manuel_rewards.png")
print("Courbe sauvegardée dans dqn_manuel_rewards.png")
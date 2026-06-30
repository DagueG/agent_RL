import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Réseau de neurones approximant Q(s, a)."""

    def __init__(self, n_observations: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # pas d'activation : sortie = Q-values


class ReplayBuffer:
    """File circulaire stockant des transitions (s, a, r, s', done)."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = DQN(n_obs, n_actions)
    buffer = ReplayBuffer(capacity=10000)

    print("Architecture du réseau DQN :")
    print(net)
    print(f"\nNombre de paramètres : {sum(p.numel() for p in net.parameters())}")
    print(f"Entrée : {n_obs} | Sortie : {n_actions}")

    # Petit test : un état aléatoire à travers le réseau
    state, _ = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = net(state_tensor)
    print(f"\nQ-values prédites pour un état aléatoire : {q_values.detach().numpy()}")

    # Petit test du buffer
    buffer.push(state, 0, 1.0, state, False)
    print(f"Taille du buffer après 1 push : {len(buffer)}")

    env.close()
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# -------------------------------
# Rede Actor-Critic
# -------------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x).squeeze(-1)


# -------------------------------
# Ambiente e modelo
# -------------------------------

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0003)


# -------------------------------
# GAE
# -------------------------------

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0

    for i in reversed(range(len(rewards))):
        next_value = 0 if i == len(rewards) - 1 else values[i + 1]

        delta = (
            rewards[i]
            + gamma * next_value * (1 - dones[i])
            - values[i]
        )

        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages, dtype=torch.float32)


# -------------------------------
# Treinamento PPO
# -------------------------------

for episode in range(200):

    state = env.reset()[0]
    done = False

    states = []
    actions = []
    rewards = []
    old_log_probs = []
    values = []
    dones = []

    while not done:

        state_tensor = torch.FloatTensor(state)

        probs, value = model(state_tensor)
        dist = Categorical(probs)

        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(
            action.item()
        )

        done = terminated or truncated

        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        old_log_probs.append(dist.log_prob(action).detach())
        values.append(value.detach().item())
        dones.append(done)

        state = next_state


    # -------------------------------
    # Calcula vantagem e retorno
    # -------------------------------

    advantages = compute_gae(rewards, values, dones)
    returns = advantages + torch.tensor(values)

    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )


    # -------------------------------
    # Recalcula a política atual
    # -------------------------------

    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)

    probs, new_values = model(states)
    dist = Categorical(probs)

    new_log_probs = dist.log_prob(actions)

    ratio = torch.exp(new_log_probs - old_log_probs)


    # -------------------------------
    # PPO Actor Loss
    # -------------------------------

    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

    actor_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()


    # -------------------------------
    # Critic Loss
    # -------------------------------

    critic_loss = (new_values - returns).pow(2).mean()


    # -------------------------------
    # Atualização
    # -------------------------------

    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if episode % 20 == 0:
        print(
            f"Episode {episode}, Reward: {sum(rewards):.0f}"
        )
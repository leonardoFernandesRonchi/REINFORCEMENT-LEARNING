import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -------------------------------
# Rede neural para o Actor-Critic
# -------------------------------
# PPO geralmente usa uma rede única que gera tanto a política (actor)
# quanto o valor do estado (critic).
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        # Camada para política (probabilidades das ações)
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Camada para valor do estado
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# -------------------------------
# Inicialização do ambiente e rede
# -------------------------------
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Função para calcular vantagem (GAE simplificado)
# -------------------------------
def compute_advantage(rewards, values, gamma=0.99):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (values[i+1] if i+1 < len(values) else 0) - values[i]
        gae = delta + gamma * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32)

# -------------------------------
# Loop de treinamento PPO
# -------------------------------
for episode in range(200):
    state = env.reset()[0]
    done = False
    rewards, log_probs, values, states, actions = [], [], [], [], []

    while not done:
        state_tensor = torch.FloatTensor(state)
        probs, value = model(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()

        next_state, reward, done, _, _ = env.step(action.item())

        # Guardamos informações para atualização
        log_probs.append(dist.log_prob(action))
        values.append(value.item())
        rewards.append(reward)
        states.append(state_tensor)
        actions.append(action)

        state = next_stateS

    # -------------------------------
    # Atualização PPO
    # -------------------------------
    advantages = compute_advantage(rewards, values)
    returns = advantages + torch.tensor(values, dtype=torch.float32)

    # Calcula perda do Actor (clipped surrogate objective)
    log_probs_tensor = torch.stack(log_probs)
    ratio = torch.exp(log_probs_tensor - log_probs_tensor.detach())
    actor_loss = -torch.min(
        ratio * advantages,
        torch.clamp(ratio, 0.8, 1.2) * advantages
    ).mean()

    # Calcula perda do Critic (erro quadrático)
    critic_loss = (torch.tensor(values, dtype=torch.float32) - returns).pow(2).mean()

    # Combina perdas
    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 20 == 0:
        print(f"Episode {episode}, Reward: {sum(rewards)}")
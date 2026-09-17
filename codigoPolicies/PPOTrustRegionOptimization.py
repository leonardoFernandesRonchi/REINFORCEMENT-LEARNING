import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -------------------------------
# Rede neural para a política
# -------------------------------

# Recebe o estado e devolve uma distribuição
# de probabilidade sobre as ações.
class PolicyNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # Probabilidades das ações
        )

    def forward(self, x):
        return self.fc(x)


# -------------------------------
# Função para calcular Fisher
# Information Matrix (FIM)
# -------------------------------

# Essa matriz mede a "curvatura" do espaço de políticas.
# Usamos ela para ajustar o gradiente de forma natural.
def fisher_information_matrix(log_probs):

    # Aproximação simples: matriz diagonal com variâncias
    probs = torch.exp(log_probs)

    return torch.diag(probs * (1 - probs))


# -------------------------------
# Inicialização do ambiente e política
# -------------------------------

env = gym.make("CartPole-v1")

policy = PolicyNet(
    env.observation_space.shape[0],
    env.action_space.n
)

optimizer = optim.Adam(policy.parameters(), lr=0.01)


# -------------------------------
# Loop de treinamento
# -------------------------------

for episode in range(200):

    state = env.reset()[0]

    rewards, log_probs = [], []

    done = False

    while not done:

        state_tensor = torch.FloatTensor(state)

        probs = policy(state_tensor)

        dist = Categorical(probs)

        action = dist.sample()

        next_state, reward, done, _, _ = env.step(
            action.item()
        )

        log_probs.append(dist.log_prob(action))

        rewards.append(reward)

        state = next_state

    # -------------------------------
    # Atualização com Natural Policy Gradient
    # -------------------------------

    total_reward = sum(rewards)

    # Gradiente padrão
    loss = -torch.stack(log_probs).sum() * total_reward

    optimizer.zero_grad()
    loss.backward()

    # Ajuste com Fisher Information Matrix
    # (natural gradient)
    with torch.no_grad():

        for param in policy.parameters():

            if param.grad is not None:

                # Multiplica pelo inverso da FIM
                # (aqui simplificado)
                fim = fisher_information_matrix(
                    torch.stack(log_probs)
                )

                # Usamos apenas a diagonal como aproximação
                natural_grad = torch.matmul(
                    torch.inverse(fim),
                    param.grad.view(-1, 1)
                ).view(param.grad.shape)

                param.grad.copy_(natural_grad)

    optimizer.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
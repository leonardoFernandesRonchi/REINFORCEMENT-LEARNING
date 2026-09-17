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

# Softmax transforma os valores anteriores

# em probabilidades.

#

# As probabilidades ficam entre 0 e 1

# e a soma delas será igual a 1.

#

# Exemplo:

# [2.1, 0.5, 1.4]

#       ↓

# [0.58, 0.12, 0.30]

#

# Ou seja:

# Ação 1 -> 58%

# Ação 2 -> 12%

# Ação 3 -> 30%

    def forward(self, x):

        return self.fc(x)


# -------------------------------

# Função para calcular Fisher

# Information Matrix (FIM)

# -------------------------------

# Essa matriz mede a "curvatura" do espaço de políticas.

# Usamos ela para ajustar o gradiente de forma natural.

def fisher_information_matrix(log_probs, policy):

    # Calcula os gradientes dos log-probs

    # em relação aos parâmetros da política.

    grads = []

    for log_prob in log_probs:

        grad = torch.autograd.grad(

            log_prob,

            policy.parameters(),

            retain_graph=True

        )

        grads.append(

            torch.cat([g.reshape(-1) for g in grad])

        )

    grads = torch.stack(grads)

    # Aproximação simples da Fisher:

    # F = média dos gradientes externos.

    fim = grads.T @ grads / len(log_probs)

    return fim


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

# traduz os numeros para probabilidades tipo 0.2 vira 20%

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

# O torch.stack junta uma sequência de tensores criando uma nova dimensão (eixo) para empilhá-los.

    loss = -torch.stack(log_probs).sum() * total_reward

    optimizer.zero_grad()

    loss.backward()

# Ajuste com Fisher Information Matrix

# (natural gradient)

    # Guarda o gradiente normal

    gradient = torch.cat([

        param.grad.reshape(-1)

        for param in policy.parameters()

        if param.grad is not None

    ])

    # Calcula a Fisher Information Matrix

    fim = fisher_information_matrix(

        log_probs,

        policy

    )

    # Pequeno valor para evitar problemas

    # ao calcular a inversa da matriz.

    fim = fim + 1e-5 * torch.eye(

        fim.size(0)

    )

# Multiplica pelo inverso da FIM

# (aqui simplificado)

    natural_grad = torch.linalg.solve(

        fim,

        gradient

    )

    # Coloca o Natural Gradient de volta

    # nos parâmetros da rede.

    index = 0

    with torch.no_grad():

        for param in policy.parameters():

            if param.grad is not None:

                size = param.numel()

                param.grad.copy_(

                    natural_grad[index:index + size]

                    .view_as(param)

                )

                index += size

    optimizer.step()

    if episode % 50 == 0:

        print(

            f"Episode {episode}, Reward: {total_reward}"

        )
import gym

import torch

import torch.nn as nn

import torch.optim as optim

from torch.distributions import Categorical


# Definimos uma rede neural que será a "policy" (política).

# Ela recebe o estado do ambiente e devolve uma distribuição de probabilidade sobre as ações.

class PolicyNet(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(PolicyNet, self).__init__()

        self.fc = nn.Sequential(

            nn.Linear(state_dim, 128),   # Primeira camada totalmente conectada

            nn.ReLU(),                   # Função de ativação ReLU

            nn.Linear(128, action_dim),  # Saída com número de ações possíveis

            nn.Softmax(dim=-1)           # Converte em probabilidades (soma = 1)

        )

    def forward(self, x):

        return self.fc(x)


# Criamos o ambiente CartPole

# (um carrinho com uma haste que deve ser equilibrada).

env = gym.make("CartPole-v1")


# Inicializamos a política com dimensões de entrada (estado) e saída (ações).

policy = PolicyNet(

    env.observation_space.shape[0],

    env.action_space.n

)


# Usamos o otimizador Adam para atualizar os pesos da rede neural.

optimizer = optim.Adam(policy.parameters(), lr=0.01)


# Loop de treinamento: o agente joga vários episódios.

for episode in range(500):

    state = env.reset()[0]   # Resetamos o ambiente no início de cada episódio

    rewards, log_probs = [], []

    done = False


    # Loop dentro de um episódio (até o jogo terminar).

    while not done:

        state_tensor = torch.FloatTensor(state)   # Converte estado em tensor

        probs = policy(state_tensor)              # Calcula probabilidades das ações

        dist = Categorical(probs)                 # Cria distribuição categórica

        action = dist.sample()                    # Escolhe ação aleatória seguindo a distribuição


        # Executa a ação no ambiente

        next_state, reward, terminated, truncated, _ = env.step(action.item())

        done = terminated or truncated


        # Guardamos o log da probabilidade da ação escolhida

        # (para calcular gradiente depois)

        log_probs.append(dist.log_prob(action))


        # Guardamos a recompensa recebida

        rewards.append(reward)


        # Atualizamos o estado

        state = next_state


    # Quando o episódio termina, calculamos o retorno de cada ação

    returns = []

    G = 0

    gamma = 0.99

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.insert(0, G)


    returns = torch.tensor(returns, dtype=torch.float32)


    # Normalizamos os retornos para deixar o treinamento mais estável

    returns = (

        returns - returns.mean()

    ) / (returns.std() + 1e-8)


    # Calculamos a perda (loss) do Policy Gradient:

    # Queremos maximizar a recompensa, então usamos sinal negativo

    # para o otimizador minimizar.

    loss = -(

        torch.stack(log_probs) * returns

    ).sum()


    # Atualizamos os parâmetros da rede neural

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    # A cada 50 episódios mostramos a recompensa total

    if episode % 50 == 0:

        print(f"Episode {episode}, Reward: {sum(rewards)}")
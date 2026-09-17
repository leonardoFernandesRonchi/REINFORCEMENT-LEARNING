import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -------------------------------
# Rede neural para o Actor (policy)
# -------------------------------

# O Actor recebe o estado e devolve uma distribuição
# de probabilidade sobre as ações.
# Ele decide qual ação tomar com base nessa distribuição.

# ============================================================
# REDE NEURAL DO ACTOR
# ============================================================
# O Actor recebe o estado atual do ambiente e determina
# a probabilidade de escolher cada ação possível.
#
# Exemplo:
# Estado -> [vida, inimigos, distância]
# Saída  -> [0.6, 0.1, 0.3]
#
# Isso significa:
# Ação 1 -> 60%
# Ação 2 -> 10%
# Ação 3 -> 30%
# ============================================================

class ActorNet(nn.Module):

    def __init__(self, state_dim, action_dim):

        # Inicializa a classe nn.Module do PyTorch.
        # Isso permite que o PyTorch controle os parâmetros
        # e o treinamento da rede neural.
        super(ActorNet, self).__init__()

        # nn.Sequential permite colocar várias camadas
        # em sequência. A saída de uma camada é enviada
        # automaticamente para a próxima.
        self.fc = nn.Sequential(

            # Camada totalmente conectada (Linear).
            #
            # state_dim = quantidade de informações recebidas
            # pelo Actor no estado.
            #
            # 128 = quantidade de neurônios dessa camada.
            #
            # Exemplo:
            # [vida, inimigos, distância]
            #          ↓
            #       Linear
            #          ↓
            #     [128 valores]
            nn.Linear(state_dim, 128),

            # Função de ativação ReLU.
            #
            # Valores negativos são transformados em 0,
            # enquanto valores positivos são mantidos.
            #
            # Exemplo:
            # [-2, 3, -1, 5] -> [0, 3, 0, 5]
            #
            # A ReLU adiciona não-linearidade à rede,
            # permitindo que ela aprenda relações mais complexas.
            nn.ReLU(),

            # Segunda camada Linear.
            #
            # Recebe os 128 valores da camada anterior
            # e produz um valor para cada ação possível.
            #
            # Se existirem 3 ações:
            # [128 valores] -> [2.1, 0.5, 1.4]
            #
            # Esses valores ainda não são probabilidades.
            nn.Linear(128, action_dim),

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
            nn.Softmax(dim=-1)
        )

    def forward(self, x):

        # Define como os dados passam pela rede.
        #
        # "x" representa o estado atual do ambiente.
        #
        # O estado passa por todas as camadas de self.fc
        # e retorna as probabilidades das ações.
        return self.fc(x)


# ============================================================
# REDE NEURAL DO CRITIC
# ============================================================
# O Critic recebe o estado atual e estima o seu valor.
#
# Diferente do Actor, ele não escolhe uma ação.
#
# Exemplo:
#
# Estado -> [vida, inimigos, distância]
#           ↓
#        Critic
#           ↓
#         7.5
#
# O valor 7.5 representa a estimativa de quão bom
# ou promissor é aquele estado.
# ============================================================

class CriticNet(nn.Module):

    def __init__(self, state_dim):

        # Inicializa a classe nn.Module do PyTorch.
        super(CriticNet, self).__init__()

        # Cria as camadas da rede do Critic em sequência.
        self.fc = nn.Sequential(

            # Recebe o estado do ambiente e transforma
            # as informações em 128 valores internos.
            #
            # Exemplo:
            # [vida, inimigos, distância]
            #          ↓
            #       Linear
            #          ↓
            #     [128 valores]
            nn.Linear(state_dim, 128),

            # Função de ativação que transforma valores
            # negativos em 0 e mantém os positivos.
            #
            # Adiciona não-linearidade à rede.
            nn.ReLU(),

            # Camada final do Critic.
            #
            # Recebe os 128 valores e produz apenas
            # um número.
            #
            # Esse único número representa a estimativa
            # do valor do estado.
            #
            # Exemplo:
            # [128 valores] -> [7.5]
            nn.Linear(128, 1)
        )

    def forward(self, x):

        # Recebe o estado "x", passa por todas as camadas
        # do Critic e retorna o valor estimado daquele estado.
        return self.fc(x)


# -------------------------------
# Inicialização do ambiente e redes
# -------------------------------

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = ActorNet(state_dim, action_dim)
critic = CriticNet(state_dim)

# Otimizadores para Actor e Critic
actor_optimizer = optim.Adam(actor.parameters(), lr=0.01)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.01)


# -------------------------------
# Loop de treinamento
# -------------------------------

for episode in range(300):

    state = env.reset()[0]   # Reset do ambiente

    done = False
    total_reward = 0

    while not done:

        state_tensor = torch.FloatTensor(state)

        # Actor escolhe ação com base nas probabilidades
        probs = actor(state_tensor)

        # Cria uma distribuição categórica a partir das probabilidades
        # exemplo: probs = [0.6, 0.1, 0.3] -> dist = Categorical(probs)
        # resulta em 60%, 10% e 30% de chance de escolher cada ação.
        dist = Categorical(probs)   # Distribuição categórica

        action = dist.sample()      # Amostra uma ação

        # Executa ação no ambiente
        next_state, reward, done, _, _ = env.step(action.item())

        # Critic avalia estado atual e próximo
        value = critic(state_tensor)                     
        next_value = critic(
            torch.FloatTensor(next_state)
        )

        # -------------------------------
        # TD Error (diferença temporal)
        # -------------------------------

        # td_target = recompensa + desconto * valor futuro
        td_target = reward + (
            0.99 * next_value * (1 - int(done))
        )

        td_error = td_target - value

        # -------------------------------
        # Atualização do Critic
        # -------------------------------

        # O Critic tenta minimizar o erro quadrático
        # entre valor atual e alvo.
        # pow(2) é usado paraa elevar o erro ao quadrado, e mean() para obter a média.
        critic_loss = td_error.pow(2).mean()

        # zera os gradientes do Critic
        critic_optimizer.zero_grad()
        # backward calcula os gradientes do loss em relação aos parâmetros da rede neural
        critic_loss.backward()
        # step atualiza os parâmetros da rede neural com base nos gradientes calculados
        critic_optimizer.step()

        # -------------------------------
        # Atualização do Actor
        # -------------------------------

        # O Actor ajusta a política para favorecer
        # ações com vantagem positiva.
        actor_loss = (
            # - inverte o sinal, e log_prob dá a probabilidade logarítmica da ação escolhida
            -dist.log_prob(action)

            #td_error representa o erro temporal do critic
            #detach -> se esse valor para calcular o Actor, mas não deixe o gradiente passar por ele até o Critic.
            * td_error.detach()
        )

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Atualiza estado e acumula recompensa
        state = next_state
        total_reward += reward

    # Mostra progresso a cada 50 episódios
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
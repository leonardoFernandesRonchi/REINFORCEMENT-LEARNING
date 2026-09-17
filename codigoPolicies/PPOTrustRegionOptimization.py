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
        # Cria a rede Actor-Critic.
        super().__init__()

        # Recebe o estado e transforma ele em 128 características.
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),

            # Ajuda a rede a aprender relações mais complexas.
            nn.ReLU()
        )

        # Parte Actor: recebe as características e calcula
        # a probabilidade de cada ação.
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),

            # Transforma os valores em probabilidades que somam 1.
            nn.Softmax(dim=-1)
        )

        # Parte Critic: recebe as mesmas características
        # e calcula o valor estimado daquele estado.
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        # Passa o estado pelas camadas iniciais.
        x = self.fc(x)

        # O Actor escolhe probabilidades para as ações
        # e o Critic estima o valor do estado.
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

    # Lista que vai guardar a vantagem (advantage) de cada estado
    advantages = []

    # Começamos o GAE com 0
    # Ele será atualizado enquanto percorremos os passos de trás para frente
    gae = 0

    # Percorremos os passos do episódio DE TRÁS PARA FRENTE
    # Exemplo: 0, 1, 2, 3, 4 -> 4, 3, 2, 1, 0
    for i in reversed(range(len(rewards))):

        # Pegamos o valor estimado pelo Critic para o próximo estado
        #
        # Se estamos no último passo, não existe próximo estado
        # dentro do episódio, então usamos 0
        next_value = 0 if i == len(rewards) - 1 else values[i + 1]

        # Calculamos o TD Error (delta)
        #
        # Ele mede a diferença entre:
        #   o que realmente aconteceu
        #   e o que o Critic esperava
        #
        # Fórmula:
        # delta = recompensa
        #       + valor do próximo estado
        #       - valor do estado atual
        #
        # gamma controla a importância do futuro
        # (1 - dones[i]) impede usar o próximo estado
        # quando o episódio já terminou
        delta = (
            rewards[i]
            + gamma * next_value * (1 - dones[i])
            - values[i]
        )

        # Atualizamos o GAE
        #
        # O GAE não considera somente o delta atual.
        # Ele também aproveita os deltas dos próximos passos.
        #
        # gamma -> desconto das recompensas futuras
        # lam   -> controla quanto dos erros futuros será utilizado
        #
        # Se o episódio terminou (done = 1), o GAE não continua
        # passando informação para trás.
        gae = delta + gamma * lam * (1 - dones[i]) * gae

        # Como estamos percorrendo de trás para frente,
        # colocamos o novo advantage no começo da lista
        # para manter a ordem original.
        advantages.insert(0, gae)

    # Transformamos a lista de advantages em um Tensor do PyTorch
    # para poder utilizá-la durante o treinamento da rede neural.
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
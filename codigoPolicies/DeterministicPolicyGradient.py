import gym
import torch
import torch.nn as nn
import torch.optim as optim

# Rede neural que representa a política determinística
# Ela recebe o estado e devolve diretamente uma ação (sem probabilidades).
class PolicyNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),   # Primeira camada
            nn.ReLU(),
            nn.Linear(128, action_dim),  # Saída com dimensão da ação
            nn.Tanh()                    # Garante que a ação fique entre -1 e 1
        )

    def forward(self, x):
        return self.fc(x)


# Rede neural para o "Critic"
# (avalia o valor Q de estado-ação)
# retorna um valor escalar que indica quão boa é a ação tomada no estado dado.
class CriticNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Valor Q
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)


# Ambiente contínuo Pendulum
env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy = PolicyNet(state_dim, action_dim)
critic = CriticNet(state_dim, action_dim)

policy_optimizer = optim.Adam(policy.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)


# Função para converter ação da rede para o espaço do ambiente
def select_action(state):

    state_tensor = torch.FloatTensor(state)

    action = policy(state_tensor)

    # Multiplicamos pelo limite máximo da ação do ambiente
    return action.detach().numpy() * env.action_space.high[0]


# Treinamento simples
for episode in range(200):

    state = env.reset()[0]

    done = False
    total_reward = 0

    while not done:

        # Seleciona ação determinística
        action = select_action(state)

        # Executa ação no ambiente
        next_state, reward, done, _, _ = env.step(action)

        # Converte para tensores
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        action_tensor = torch.FloatTensor(action)

        # Calcula valor Q atual
        q_value = critic(state_tensor, action_tensor)

        # Calcula valor Q alvo (reward + valor futuro)
        with torch.no_grad():

            next_action = policy(next_state_tensor)

            q_target = reward + 0.99 * critic(
                next_state_tensor,
                next_action
            )

        # Atualiza Critic
        # (minimiza erro entre Q atual e Q alvo)
        # pow(2) é usado paraa elevar o erro ao quadrado, e mean() para obter a média.
        critic_loss = (q_value - q_target).pow(2).mean()

        #zero_grad() limpa os gradientes
        critic_optimizer.zero_grad()

        #backward() calcula os gradientes do loss em relação aos parâmetros da rede neural
        critic_loss.backward()

        #step() atualiza os parâmetros da rede neural com base nos gradientes calculados
        critic_optimizer.step()

        # Atualiza Policy
        # (maximiza valor Q)
        policy_loss = -critic(
            state_tensor,
            policy(state_tensor)
        ).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        state = next_state

        total_reward += reward

    if episode % 20 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
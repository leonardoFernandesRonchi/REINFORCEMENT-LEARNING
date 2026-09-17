import gym

import torch

import torch.nn as nn

import torch.optim as optim

import random

from collections import deque


# -------------------------------

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


# -------------------------------

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


# -------------------------------

# Replay Buffer

# -------------------------------

# Guarda experiências anteriores para treinar

# a rede usando diferentes experiências.

class ReplayBuffer:

    def __init__(self, capacity=10000):

        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):

        self.buffer.append(

            (state, action, reward, next_state, done)

        )

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (

            torch.FloatTensor(states),

            torch.FloatTensor(actions),

            torch.FloatTensor(rewards).unsqueeze(1),

            torch.FloatTensor(next_states),

            torch.FloatTensor(dones).unsqueeze(1)

        )

    def __len__(self):

        return len(self.buffer)


# -------------------------------

# Ambiente contínuo Pendulum

# -------------------------------

env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]

action_dim = env.action_space.shape[0]

policy = PolicyNet(state_dim, action_dim)

critic = CriticNet(state_dim, action_dim)


# -------------------------------

# Redes Target

# -------------------------------

# São cópias das redes principais usadas

# para deixar o treinamento mais estável.

target_policy = PolicyNet(state_dim, action_dim)

target_critic = CriticNet(state_dim, action_dim)

target_policy.load_state_dict(policy.state_dict())

target_critic.load_state_dict(critic.state_dict())


policy_optimizer = optim.Adam(policy.parameters(), lr=0.001)

critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

buffer = ReplayBuffer()

gamma = 0.99

tau = 0.005


# -------------------------------

# Função para converter ação da rede para o espaço do ambiente

# -------------------------------

def select_action(state):

    state_tensor = torch.FloatTensor(state)

    action = policy(state_tensor)

    return action.detach().numpy() * env.action_space.high[0]


# -------------------------------

# Treinamento simples

# -------------------------------

for episode in range(200):

    state = env.reset()[0]

    done = False

    total_reward = 0

    while not done:

        # Seleciona ação determinística

        action = select_action(state)

        # Adiciona ruído para exploração

        action += 0.1 * torch.randn(action_dim).numpy()

        action = action.clip(

            env.action_space.low,

            env.action_space.high

        )

        # Executa ação no ambiente

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Guarda experiência no Replay Buffer

        buffer.add(

            state,

            action,

            reward,

            next_state,

            done

        )

        # -------------------------------

        # Atualização usando Replay Buffer

        # -------------------------------

        if len(buffer) >= 64:

            states, actions, rewards, next_states, dones = buffer.sample(64)

            # Calcula valor Q alvo usando as redes Target

            with torch.no_grad():

                next_actions = target_policy(next_states)

                next_actions = next_actions * env.action_space.high[0]

                next_q = target_critic(

                    next_states,

                    next_actions

                )

                q_target = rewards + gamma * next_q * (1 - dones)

            # Calcula valor Q atual

            q_value = critic(states, actions)

            # Atualiza Critic

            # (minimiza erro entre Q atual e Q alvo)

            critic_loss = (

                q_value - q_target

            ).pow(2).mean()

            # zero_grad() limpa os gradientes

            critic_optimizer.zero_grad()

            # backward() calcula os gradientes

            critic_loss.backward()

            # step() atualiza os parâmetros

            critic_optimizer.step()

            # -------------------------------

            # Atualiza Policy

            # -------------------------------

            # (maximiza valor Q)

            policy_actions = policy(states)

            policy_actions = (

                policy_actions * env.action_space.high[0]

            )

            policy_loss = -critic(

                states,

                policy_actions

            ).mean()

            policy_optimizer.zero_grad()

            policy_loss.backward()

            policy_optimizer.step()

            # -------------------------------

            # Atualização suave das redes Target

            # -------------------------------

            for target_param, param in zip(

                target_policy.parameters(),

                policy.parameters()

            ):

                target_param.data.copy_(

                    tau * param.data +

                    (1 - tau) * target_param.data

                )

            for target_param, param in zip(

                target_critic.parameters(),

                critic.parameters()

            ):

                target_param.data.copy_(

                    tau * param.data +

                    (1 - tau) * target_param.data

                )

        state = next_state

        total_reward += reward

    if episode % 20 == 0:

        print(

            f"Episode {episode}, Reward: {total_reward}"

        )
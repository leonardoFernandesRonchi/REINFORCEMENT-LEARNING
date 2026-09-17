import gym

import torch

import torch.nn as nn

import torch.optim as optim

from torch.distributions import Categorical


# -------------------------------

# Actor: Rede neural para a política

# -------------------------------

class ActorNet(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(ActorNet, self).__init__()

        self.fc = nn.Sequential(

            nn.Linear(state_dim, 128),

            nn.ReLU(),

            nn.Linear(128, action_dim),

            nn.Softmax(dim=-1)  # Probabilidades das ações

        )

    def forward(self, x):

        return self.fc(x)


# -------------------------------

# Critic: Rede neural para Q-learning

# -------------------------------

# Em vez de apenas valor do estado,

# aqui o Critic estima Q(s,a).

class CriticNet(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(CriticNet, self).__init__()

        self.fc = nn.Sequential(

            nn.Linear(state_dim, 128),

            nn.ReLU(),

            nn.Linear(128, action_dim)  # Q-value para cada ação

        )

    def forward(self, x):

        return self.fc(x)


# -------------------------------

# Inicialização do ambiente e redes

# -------------------------------

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]

action_dim = env.action_space.n

actor = ActorNet(state_dim, action_dim)

critic = CriticNet(state_dim, action_dim)

actor_optimizer = optim.Adam(

    actor.parameters(),

    lr=0.01

)

critic_optimizer = optim.Adam(

    critic.parameters(),

    lr=0.01

)

gamma = 0.99  # fator de desconto


# -------------------------------

# Loop de treinamento

# -------------------------------

for episode in range(300):

    state = env.reset()[0]

    done = False

    total_reward = 0

    while not done:

        state_tensor = torch.FloatTensor(state)

        # Actor escolhe ação com base nas probabilidades

        probs = actor(state_tensor)

        dist = Categorical(probs)

        action = dist.sample()

        # Executa ação no ambiente

        next_state, reward, terminated, truncated, _ = env.step(

            action.item()

        )

        done = terminated or truncated

        next_state_tensor = torch.FloatTensor(

            next_state

        )

        # -------------------------------

        # Atualização do Critic (Q-learning)

        # -------------------------------

        q_values = critic(state_tensor)

        q_value = q_values[action]

        with torch.no_grad():

            next_q_values = critic(

                next_state_tensor

            )

            max_next_q = torch.max(next_q_values)

            q_target = (

                reward

                + gamma * max_next_q

                * (1 - int(done))

            )

        critic_loss = (

            q_value - q_target

        ).pow(2).mean()

        critic_optimizer.zero_grad()

        critic_loss.backward()

        critic_optimizer.step()

        # -------------------------------

        # Atualização do Actor (Policy Gradient)

        # -------------------------------

        # O Actor ajusta a política para favorecer

        # ações com maior vantagem.

        q_values = critic(state_tensor)

        with torch.no_grad():

            value = (

                probs * q_values

            ).sum()

            advantage = (

                q_values[action] - value

            )

        actor_loss = (

            -dist.log_prob(action)

            * advantage

        )

        actor_optimizer.zero_grad()

        actor_loss.backward()

        actor_optimizer.step()

        state = next_state

        total_reward += reward

    if episode % 50 == 0:

        print(

            f"Episode {episode}, Reward: {total_reward}"

        )
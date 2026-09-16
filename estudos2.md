# Resumo de Métodos de Policy Gradient em RL

## 📖 O que é uma _Policy_

- **[Policy](ca://s?q=O_que_e_policy_em_Reinforcement_Learning)** é uma função que mapeia estados para ações.
- Pode ser **determinística** (sempre escolhe a mesma ação para um estado) ou **estocástica** (define uma distribuição de probabilidade sobre ações).

## 🎯 O que é _Policy Gradient_

- **[Policy Gradient](ca://s?q=O_que_e_policy_gradient)** é uma família de algoritmos que ajusta diretamente os parâmetros da _policy_ para maximizar a recompensa esperada.
- Em vez de aprender um valor de ação (Q-value), aprende-se a própria política.

---

## 🌐 Stochastic Policy Gradient

- Usa políticas **estocásticas** (ex.: softmax sobre ações).
- Bom para ambientes com alta incerteza ou quando explorar é crucial.
- **Não usar** quando o espaço de ação é contínuo e determinístico (ineficiente).

## ⚡ Deterministic Policy Gradient (DPG)

- Usa políticas **determinísticas** (ex.: ação = função direta do estado).
- Mais eficiente em espaços contínuos de ação.
- **Não usar** em ambientes onde a aleatoriedade é essencial para explorar.

## 🎭 Actor-Critic Methods

- Combina um **actor** (policy) e um **critic** (value function).
- Reduz variância do gradiente e melhora estabilidade.
- Útil em ambientes complexos e contínuos.
- **Não usar** se o ambiente é simples e tabular (overkill).

## 🌱 Natural Policy Gradients

- Ajusta o gradiente levando em conta a geometria da distribuição de políticas.
- Mais estável e eficiente que gradiente padrão.
- **Não usar** em problemas pequenos, pois é mais caro computacionalmente.

## 🔒 Trust Region Optimization (TRPO/PPO)

- Impõe restrições para que a atualização da política não seja muito grande.
- Evita instabilidade e colapso da política.
- **Não usar** em problemas simples onde atualizações grandes não causam problemas.

## 🔀 Combining Policy Gradient and Q-learning

- Usa Q-learning para estimar valores e policy gradient para otimizar a política.
- Útil em ambientes com grandes espaços de ação.
- **Não usar** se o ambiente é pequeno e tabular (complexidade desnecessária).

---

# Exemplo de Stochastic Policy Gradient com Gym (comentado)

Este código mostra como treinar um agente usando **[Stochastic Policy Gradient](ca://s?q=Exemplo_de_stochastic_policy_gradient)** no ambiente `CartPole-v1` do Gym.  
Os comentários explicam cada parte para alguém que está começando em **[Reinforcement Learning](ca://s?q=Introducao_a_Reinforcement_Learning)**.

```python
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

# Criamos o ambiente CartPole (um carrinho com uma haste que deve ser equilibrada).
env = gym.make("CartPole-v1")

# Inicializamos a política com dimensões de entrada (estado) e saída (ações).
policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)

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
        next_state, reward, done, _, _ = env.step(action.item())

        # Guardamos o log da probabilidade da ação escolhida (para calcular gradiente depois)
        log_probs.append(dist.log_prob(action))
        # Guardamos a recompensa recebida
        rewards.append(reward)

        # Atualizamos o estado
        state = next_state

    # Quando o episódio termina, calculamos a recompensa total
    total_reward = sum(rewards)

    # Calculamos a perda (loss) do Policy Gradient:
    # Queremos maximizar a recompensa, então usamos sinal negativo para o otimizador minimizar.
    loss = -torch.stack(log_probs).sum() * total_reward

    # Atualizamos os parâmetros da rede neural
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # A cada 50 episódios mostramos a recompensa total
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
```

# Exemplo de Deterministic Policy Gradient (DPG) com Gym

O **[Deterministic Policy Gradient](ca://s?q=Exemplo_de_deterministic_policy_gradient)** é usado em ambientes com **ações contínuas**.  
Diferente do _stochastic policy gradient_, aqui a política não gera uma distribuição de probabilidades, mas sim uma **ação direta** (determinística) para cada estado.

No exemplo abaixo, usamos o ambiente `Pendulum-v1` do Gym, que possui espaço de ação contínuo.  
Os comentários explicam cada parte para iniciantes em **[Reinforcement Learning](ca://s?q=Introducao_a_Reinforcement_Learning)**.

```python
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
            nn.Linear(128, action_dim), # Saída com dimensão da ação
            nn.Tanh()                   # Garante que a ação fique entre -1 e 1
        )

    def forward(self, x):
        return self.fc(x)

# Rede neural para o "Critic" (avalia o valor Q de estado-ação)
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
            q_target = reward + 0.99 * critic(next_state_tensor, next_action)

        # Atualiza Critic (minimiza erro entre Q atual e Q alvo)
        critic_loss = (q_value - q_target).pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Atualiza Policy (maximiza valor Q esperado)
        policy_loss = -critic(state_tensor, policy(state_tensor)).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        state = next_state
        total_reward += reward

    if episode % 20 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")

```

# Exemplo de Actor-Critic Methods com Gym (comentado)

Os **[Actor-Critic Methods](ca://s?q=Exemplo_de_actor_critic_com_PyTorch)** combinam:

- **Actor**: a política que escolhe ações.
- **Critic**: uma função de valor que avalia quão boas foram essas ações.

Isso ajuda a reduzir a variância e estabilizar o aprendizado.  
Aqui está um exemplo com o ambiente `CartPole-v1` do Gym, com comentários explicando cada parte.

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -------------------------------
# Rede neural para o Actor (policy)
# -------------------------------
# O Actor recebe o estado e devolve uma distribuição de probabilidade sobre as ações.
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),   # Primeira camada totalmente conectada
            nn.ReLU(),                   # Função de ativação
            nn.Linear(128, action_dim),  # Saída com número de ações possíveis
            nn.Softmax(dim=-1)           # Converte em probabilidades
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------------
# Rede neural para o Critic (value function)
# -------------------------------
# O Critic recebe o estado e devolve um valor escalar (estimativa de quão bom é o estado).
class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super(CriticNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Valor esperado do estado
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------------
# Inicialização do ambiente e redes
# -------------------------------
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]  # Dimensão do estado
action_dim = env.action_space.n             # Número de ações possíveis

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
        dist = Categorical(probs)   # Distribuição categórica
        action = dist.sample()      # Amostra uma ação

        # Executa ação no ambiente
        next_state, reward, done, _, _ = env.step(action.item())

        # Critic avalia estado atual e próximo
        value = critic(state_tensor)                     # Valor atual
        next_value = critic(torch.FloatTensor(next_state))  # Valor do próximo estado

        # -------------------------------
        # TD Error (diferença temporal)
        # -------------------------------
        # td_target = recompensa + desconto * valor futuro
        td_target = reward + (0.99 * next_value * (1 - int(done)))
        td_error = td_target - value

        # -------------------------------
        # Atualização do Critic
        # -------------------------------
        # O Critic tenta minimizar o erro quadrático entre valor atual e alvo.
        critic_loss = td_error.pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # -------------------------------
        # Atualização do Actor
        # -------------------------------
        # O Actor ajusta a política para favorecer ações com vantagem positiva.
        actor_loss = -dist.log_prob(action) * td_error.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Atualiza estado e acumula recompensa
        state = next_state
        total_reward += reward

    # Mostra progresso a cada 50 episódios
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
```

# Exemplo de Natural Policy Gradients com Gym (comentado)

Os **[Natural Policy Gradients](ca://s?q=Exemplo_de_natural_policy_gradients)** são uma variação dos métodos de _policy gradient_.  
A ideia é levar em conta a **geometria da distribuição de políticas**: em vez de atualizar os parâmetros com o gradiente padrão, usamos o **gradiente natural**, que ajusta a política de forma mais estável e eficiente.

Aqui está um exemplo simplificado usando `CartPole-v1` com PyTorch.  
Os comentários explicam cada parte para iniciantes.

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -------------------------------
# Rede neural para a política
# -------------------------------
# Recebe o estado e devolve uma distribuição de probabilidade sobre as ações.
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
# Função para calcular Fisher Information Matrix (FIM)
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
policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
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

        next_state, reward, done, _, _ = env.step(action.item())

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

    # Ajuste com Fisher Information Matrix (natural gradient)
    with torch.no_grad():
        for param in policy.parameters():
            if param.grad is not None:
                # Multiplica pelo inverso da FIM (aqui simplificado)
                fim = fisher_information_matrix(torch.stack(log_probs))
                # Usamos apenas a diagonal como aproximação
                natural_grad = torch.matmul(torch.inverse(fim), param.grad.view(-1,1)).view(param.grad.shape)
                param.grad.copy_(natural_grad)

    optimizer.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
```

# Exemplo de Trust Region Optimization (PPO/TRPO) com Gym (comentado)

Os métodos de **[Trust Region Optimization](ca://s?q=Exemplo_de_trust_region_optimization)**, como **TRPO** e **PPO**, são evoluções dos algoritmos de _policy gradient_.  
A ideia é **limitar o tamanho da atualização da política** para evitar que ela mude demais de um episódio para outro, o que poderia causar instabilidade.

O **PPO (Proximal Policy Optimization)** é uma versão prática e mais simples de TRPO.  
Aqui está um exemplo com `CartPole-v1` usando PyTorch, com comentários explicando cada parte.

```python
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

        state = next_state

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
```

# Exemplo de Combining Policy Gradient e Q-learning (Actor-Critic + Q-learning)

A ideia de **[Combinar Policy Gradient e Q-learning](ca://s?q=Exemplo_de_policy_gradient_com_Q_learning)** é usar:

- **Actor-Critic (Policy Gradient)**: o _actor_ aprende uma política diretamente, e o _critic_ avalia estados.
- **Q-learning**: fornece uma estimativa de valor de ação (Q-value), que pode ser usada para guiar o critic.

Assim, o critic não depende apenas de regressão direta de valores, mas também aproveita a estrutura de Q-learning para melhorar a estabilidade.

Aqui está um exemplo simplificado com `CartPole-v1` usando PyTorch, com comentários detalhados.

```python
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
# Em vez de apenas valor do estado, aqui o Critic estima Q(s,a).
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

actor_optimizer = optim.Adam(actor.parameters(), lr=0.01)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.01)

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
        next_state, reward, done, _, _ = env.step(action.item())
        next_state_tensor = torch.FloatTensor(next_state)

        # -------------------------------
        # Atualização do Critic (Q-learning)
        # -------------------------------
        q_values = critic(state_tensor)
        q_value = q_values[action]

        with torch.no_grad():
            next_q_values = critic(next_state_tensor)
            max_next_q = torch.max(next_q_values)
            q_target = reward + gamma * max_next_q * (1 - int(done))

        critic_loss = (q_value - q_target).pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # -------------------------------
        # Atualização do Actor (Policy Gradient)
        # -------------------------------
        # O Actor ajusta a política para favorecer ações com maior Q-value.
        advantage = q_target - q_value.detach()
        actor_loss = -dist.log_prob(action) * advantage
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state
        total_reward += reward

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
```

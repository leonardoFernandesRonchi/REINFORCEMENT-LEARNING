# 📚 Estudos de Aprendizado por Reforço (Deep RL)

## 📌 Índice

### 1. Definições e Conceitos Fundamentais

- [1.1. Propriedade de Markov e MDPs](#11-propriedade-de-markov-e-mdps)
- [1.2. Q-Learning Tabular](#12-q-learning-tabular)
- [1.3. Fitted Q-Learning & Deep Q-Networks (DQN)](#13-fitted-q-learning--deep-q-networks-dqn)
- [1.4. Métodos de Gradiente de Política e Actor-Critic](#14-métodos-de-gradiente-de-política-e-actor-critic)

### 2. Exemplos de Código Prático

- [💻 Código 1: Simulação com a Propriedade de Markov](#código-1-simulação-com-a-propriedade-de-markov)
- [💻 Código 2: Algoritmo Q-Learning Tabular](#código-2-algoritmo-q-learning-tabular)
- [💻 Código 3: Agente Deep Q-Network (DQN)](#código-3-agente-deep-q-network-dqn)
- [💻 Código 4: Algoritmo REINFORCE (Policy Gradient)](#código-4-algoritmo-reinforce-policy-gradient)

---

# 1. Definições e Conceitos Fundamentais

## 1.1. Propriedade de Markov e MDPs

A **Propriedade de Markov** estabelece que a probabilidade da próxima observação e recompensa no tempo `t+1` depende exclusivamente da observação e da ação tomadas no estado atual no tempo `t`, sem necessidade de consultar o histórico completo de passos anteriores [1].

Em termos simples:

> **O presente contém toda a informação necessária para decidir o próximo passo.**

Um Processo de Decisão de Markov (MDP) formaliza esse ambiente através de uma 5-upla `(S, A, T, R, γ)`, onde:

- `S` é o espaço de estados;
- `A` é o espaço de ações;
- `T` é a função de transição;
- `R` é a função de recompensa;
- `γ` é o fator de desconto.

Quando usar: quando o estado atual possui as informações necessárias para tomar decisões e o problema envolve decisões sequenciais.

Quando não usar: quando o estado atual não possui informações suficientes e é necessário considerar o histórico para tomar uma decisão.

[💻 Ir direto para o exemplo de Markov](#código-1-simulação-com-a-propriedade-de-markov)

---

## 1.2. Q-Learning Tabular

O **Q-Learning** básico mantém uma tabela de busca (_lookup table_) com uma entrada explícita para cada par estado-ação `Q(s, a)` [3].

Para aprender a função de valor ótima `Q*`, o algoritmo aplica iterativamente o **Operador de Bellman** `B` [7].

Pelo Teorema de Banach, o operador de Bellman é uma aplicação de contração, o que garante a existência de um ponto fixo único e a convergência sob amostragem repetida em espaços discretos [7].

Quando usar: quando o número de estados e ações é pequeno e discreto, como em labirintos simples e ambientes pequenos.

Quando não usar: quando o espaço de estados é muito grande ou contínuo, pois a tabela pode ficar grande demais ou inviável de ser criada.

[💻 Ir direto para o exemplo de Q-Learning](#código-2-algoritmo-q-learning-tabular)

---

## 1.3. Fitted Q-Learning & Deep Q-Networks (DQN)

Quando o espaço de estados é de alta dimensão ou contínuo, a tabela de busca torna-se inviável, exigindo o uso de uma função de valor parametrizada por uma rede neural:

`Q(s, a; θ)`

[8, 9]

O **Fitted Q-Learning** treina essa rede minimizando o erro quadrático em relação a um valor alvo:

`YₖQ = r + γ maxₐ' Q(s', a'; θₖ)`

[10, 11]

O algoritmo **Deep Q-Networks (DQN)** estabiliza esse aprendizado em ambientes complexos introduzindo duas heurísticas:

1. **Target Q-Network** — utiliza pesos congelados por `C` iterações para evitar alvos móveis.
2. **Replay Memory** — armazena experiências e realiza amostragem aleatória de lotes para quebrar a correlação sequencial dos dados.

Quando usar: quando o espaço de estados é grande ou complexo, mas as ações são discretas, como em jogos e ambientes com muitas possibilidades.

Quando não usar: quando o espaço de estados é pequeno, pois o Q-Learning Tabular pode ser mais simples. Também não é a opção mais adequada quando as ações são contínuas.

[4, 12, 13]

[💻 Ir direto para o exemplo de DQN](#código-3-agente-deep-q-network-dqn)

---

## 1.4. Métodos de Gradiente de Política e Actor-Critic

Diferente dos métodos baseados em valor, os **métodos de gradiente de política** otimizam diretamente uma política:

`πw(s, a)`

ajustando seus parâmetros `w` por subida de gradiente estocástico de acordo com o Teorema do Gradiente de Política [5].

Em arquiteturas **Actor-Critic**, o sistema é dividido em duas partes:

- **Actor** — rede que atualiza a política;
- **Critic** — rede que estima a função de valor para avaliar a qualidade das ações tomadas.

Quando usar: quando é interessante aprender diretamente a política ou quando o problema possui ações mais complexas ou contínuas.

Quando não usar: em problemas muito simples e pequenos, nos quais o Q-Learning Tabular pode resolver o problema de forma mais simples.

[💻 Ir direto para o exemplo de Policy Gradient](#código-4-algoritmo-reinforce-policy-gradient)

---

# 2. Exemplos de Código Prático

## 💻 Código 1: Simulação com a Propriedade de Markov

```python
import gymnasium as gym

# Criamos o ambiente do CartPole
env = gym.make("CartPole-v1")

state, info = env.reset()

done = False

while not done:

    # PROPRIEDADE DE MARKOV:
    # A tomada de decisão utiliza exclusivamente
    # o estado atual (state), sem guardar histórico
    # de estados passados.

    action = env.action_space.sample()

    # O ambiente calcula o próximo estado
    # baseado unicamente em (state, action)
    next_state, reward, terminated, truncated, info = env.step(action)

    state = next_state

    done = terminated or truncated

env.close()

print("Simulação concluída!")
```

[⬆️ Voltar ao índice](#-estudos-de-aprendizado-por-reforço-deep-rl) | [📖 Voltar para a definição de Markov](#11-propriedade-de-markov-e-mdps)

---

## 💻 Código 2: Algoritmo Q-Learning Tabular

```python
import gymnasium as gym
import numpy as np

# 1. CONFIGURAÇÃO DO AMBIENTE
# Cria o ambiente "FrozenLake" (Lago Congelado).
# is_slippery=False significa que o gelo não escorrega (se o agente tentar ir para a direita, ele vai para a direita).
env = gym.make("FrozenLake-v1", is_slippery=False)

# 2. INICIALIZAÇÃO DA TABELA Q
# A Tabela Q é o "cérebro" do agente. Ela mapeia qual a utilidade de tomar cada ação em cada estado.
# Criamos uma matriz cheia de zeros com o tamanho (Número de Estados x Número de Ações).
q_table = np.zeros((
    env.observation_space.n, # Linhas: todos os estados possíveis (posições no mapa)
    env.action_space.n       # Colunas: todas as ações possíveis (cima, baixo, esquerda, direita)
))

# 3. HIPERPARÂMETROS
alpha = 0.1      # Taxa de aprendizado: define o peso da nova informação em relação à antiga (10%).
gamma = 0.99     # Fator de desconto: define o quanto o agente valoriza recompensas futuras vs imediatas.
epsilon = 0.1    # Taxa de exploração: 10% de chance do agente tomar uma ação aleatória para descobrir novos caminhos.

# 4. LOOP DE TREINAMENTO (1000 partidas)
for episode in range(1000):

    # Reseta o ambiente para começar um novo episódio.
    # 'state' recebe a posição inicial. O '_' ignora informações extras que não usaremos.
    state, _ = env.reset()

    done = False # Controla se o episódio atual acabou (vitória, derrota ou limite de tempo)

    while not done:

        # 5. ESCOLHA DA AÇÃO (Estratégia Epsilon-Greedy)
        # Sorteia um número entre 0 e 1. Se for menor que 0.1, o agente "explora" (ação aleatória).
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # Caso contrário, o agente "explota" (escolhe a melhor ação que já aprendeu para este estado).
            action = np.argmax(q_table[state])

        # 6. INTERAÇÃO COM O AMBIENTE
        # O agente executa a ação e o ambiente retorna as consequências:
        next_state, reward, terminated, truncated, _ = env.step(action)

        # O episódio acaba se o agente caiu no buraco/chegou no objetivo (terminated)
        # ou se excedeu o limite de passos permitidos (truncated).
        done = terminated or truncated

        # 7. ATUALIZAÇÃO DA TABELA Q (Equação de Bellman)
        # Primeiro, olhamos para o próximo estado e vemos qual é a melhor ação possível lá.
        best_next_action = np.argmax(q_table[next_state])

        # Calculamos o "Alvo" (Target): a recompensa real recebida + a estimativa da melhor recompensa futura.
        td_target = (
            reward
            + gamma * q_table[next_state, best_next_action]
        )

        # Atualizamos o valor da ação que acabamos de tomar na Tabela Q.
        # Fórmula: Valor Antigo + Taxa de Aprendizado * (Alvo - Valor Antigo)
        q_table[state, action] += (
            alpha
            * (td_target - q_table[state, action])
        )

        # O agente avança fisicamente para o próximo estado para o próximo ciclo.
        state = next_state

# Fecha o ambiente ao terminar todo o treinamento para liberar memória.
env.close()

```

[⬆️ Voltar ao índice](#-estudos-de-aprendizado-por-reforço-deep-rl) | [📖 Voltar para a definição de Q-Learning](#12-q-learning-tabular)

---

## 💻 Código 3: Agente Deep Q-Network (DQN)

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. REDE NEURAL (O Cérebro do Agente)
# Recebe os dados do ambiente (estado) e devolve uma nota (Q-value) para cada ação possível.
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# 2. CONFIGURAÇÃO INICIAL
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# No DQN, usamos DUAS redes neurais para manter o aprendizado estável.
q_net = QNetwork(state_dim, action_dim)      # Rede Principal: joga e aprende ativamente
target_net = QNetwork(state_dim, action_dim) # Rede Alvo: usada só de referência para a meta
target_net.load_state_dict(q_net.state_dict()) # Começam idênticas

optimizer = optim.Adam(q_net.parameters(), lr=0.001)

# 3. MEMÓRIA (Replay Memory)
# Guarda os últimos 10.000 acontecimentos. O agente aprende revendo o passado.
replay_memory = deque(maxlen=10000)

# 4. HIPERPARÂMETROS
gamma = 0.99       # Fator de desconto (foco no longo prazo)
batch_size = 64    # Quantas memórias ele revê por vez ao treinar
C = 100            # Frequência (em passos) de atualização da Rede Alvo
epsilon = 0.1      # Taxa de exploração (10% de chance de agir aleatoriamente)

state, _ = env.reset()

# 5. LOOP DE INTERAÇÃO E TREINAMENTO (1000 passos)
for step in range(1, 1001):

    # Prepara o estado para a rede neural (transforma em Tensor do PyTorch)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    # Escolhe a Ação (Epsilon-Greedy)
    if random.random() < epsilon:
        action = env.action_space.sample() # Explora (ação aleatória)
    else:
        with torch.no_grad():
            # Explota (pede para a rede principal a ação com maior nota)
            action = torch.argmax(q_net(state_tensor)).item()

    # Executa a ação
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Salva a experiência na Memória
    replay_memory.append((state, action, reward, next_state, done))

    if done:
        state, _ = env.reset()
    else:
        state = next_state

    # 6. TREINAMENTO (Aprende revendo o passado)
    # Só treina se já tivermos guardado memórias suficientes (batch_size)
    if len(replay_memory) >= batch_size:

        # Pega 64 memórias aleatórias do passado
        batch = random.sample(replay_memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Converte essas memórias para Tensors (formato da rede neural)
        b_s = torch.FloatTensor(np.array(states))
        b_a = torch.LongTensor(actions).unsqueeze(1)
        b_r = torch.FloatTensor(rewards).unsqueeze(1)
        b_ns = torch.FloatTensor(np.array(next_states))
        b_d = torch.FloatTensor(dones).unsqueeze(1) # 1 se o jogo acabou, 0 se não

        # Q-Values atuais: O que a Rede Principal achava que ia ganhar
        q_values = q_net(b_s).gather(1, b_a)

        # Q-Values Alvo: Equação de Bellman
        with torch.no_grad():
            # Pergunta para a Rede Alvo qual a melhor nota do próximo estado
            max_next_q = target_net(b_ns).max(1, keepdim=True)[0]

            # Se o jogo acabou (b_d = 1), o alvo é só a recompensa.
            # Senão, soma a recompensa com o que espera ganhar no futuro.
            targets = b_r + (1 - b_d) * gamma * max_next_q

        # Calcula o Erro (Diferença entre o que a rede previu e o alvo real)
        loss = nn.MSELoss()(q_values, targets)

        # Atualiza os pesos da Rede Principal (Backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 7. ATUALIZAÇÃO DA REDE ALVO
    # A cada 'C' passos, copiamos o conhecimento da Rede Principal para a Rede Alvo
    if step % C == 0:
        target_net.load_state_dict(q_net.state_dict())

env.close()
print("DQN treinado com sucesso!")
```

[⬆️ Voltar ao índice](#-estudos-de-aprendizado-por-reforço-deep-rl) | [📖 Voltar para a definição de DQN](#13-fitted-q-learning--deep-q-networks-dqn)

---

## 💻 Código 4: Algoritmo REINFORCE (Policy Gradient)

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 1. A REDE NEURAL (A "Política" do Agente)
# Diferente do DQN que retorna um valor (nota), esta rede retorna a PROBABILIDADE de escolher cada ação.
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1) # Garante que a saída seja uma porcentagem (ex: 70% Esquerda, 30% Direita)
        )

    def forward(self, x):
        return self.fc(x)

# 2. CONFIGURAÇÃO INICIAL
env = gym.make("CartPole-v1")

# Cria a rede neural da política
policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

gamma = 0.99 # Fator de desconto (o quanto valorizamos o futuro)

# 3. LOOP DE TREINAMENTO (300 Episódios)
# Policy Gradient (REINFORCE) aprende episódios inteiros por vez (Monte Carlo), e não passo a passo.
for episode in range(300):

    state, _ = env.reset()
    log_probs = [] # Guarda as probabilidades das ações que escolhemos
    rewards = []   # Guarda as recompensas de cada passo
    done = False

    # A) JOGAR UM EPISÓDIO INTEIRO
    while not done:

        # Converte o estado para o formato do PyTorch
        state_tensor = torch.FloatTensor(state)

        # Pede para a rede as probabilidades de cada ação (ex: [0.7, 0.3])
        probs = policy_net(state_tensor)

        # Cria uma "roleta" viciada com essas probabilidades e sorteia a ação
        distribution = Categorical(probs)
        action = distribution.sample()

        # Executa a ação no ambiente
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        # Salva a probabilidade (em logaritmo) da ação que acabamos de tomar e a recompensa
        log_probs.append(distribution.log_prob(action))
        rewards.append(reward)

        state = next_state

    # B) CALCULAR O RETORNO DESCONTADO (Monte Carlo)
    # Como o jogo acabou, olhamos de trás para frente (do final pro começo).
    # A ideia é: as primeiras ações também foram responsáveis pela recompensa lá do final.
    discounted_rewards = []
    R = 0

    for r in reversed(rewards):
        R = r + gamma * R # Soma a recompensa atual com o futuro (descontado)
        discounted_rewards.insert(0, R) # Coloca no começo da lista

    # Transforma a lista numa estrutura do PyTorch
    discounted_rewards = torch.FloatTensor(discounted_rewards)

    # C) ATUALIZAR A REDE NEURAL (Gradient Ascent)
    policy_loss = []

    # Para cada passo que demos no jogo:
    for log_prob, G in zip(log_probs, discounted_rewards):
        # Fórmula do REINFORCE: -log(probabilidade) * Retorno(G)
        # O sinal de menos (-) é porque o PyTorch sempre tenta MINIMIZAR erros (Gradient Descent).
        # Como queremos MAXIMIZAR a recompensa, invertemos o sinal.
        policy_loss.append(-log_prob * G)

    optimizer.zero_grad()

    # Soma os erros de todo o episódio em um único valor
    policy_loss = torch.stack(policy_loss).sum()

    # Faz a matemática mágica (Backpropagation) para ajustar os pesos da rede
    policy_loss.backward()
    optimizer.step()

env.close()
print("Policy Gradient treinado com sucesso!")
```

[⬆️ Voltar ao índice](#-estudos-de-aprendizado-por-reforço-deep-rl) | [📖 Voltar para a definição de Policy Gradient](#14-métodos-de-gradiente-de-política-e-actor-critic)

---

# 📚 Referências

Adicione aqui as referências utilizadas nos estudos.

- [1] Propriedade de Markov
- [2] Processos de Decisão de Markov (MDP)
- [3] Q-Learning
- [4] Deep Q-Networks (DQN)
- [5] Policy Gradient
- [6] Actor-Critic
- [7] Operador de Bellman
- [8] Função de valor
- [9] Aproximação por redes neurais
- [10] Fitted Q-Learning
- [11] Target Value
- [12] Target Network
- [13] Experience Replay

[⬆️ Voltar ao índice](#-estudos-de-aprendizado-por-reforço-deep-rl)

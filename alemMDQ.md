# 📚 Capítulo 10: Aprendizado por Reforço Profundo Além dos MDPs

## 📌 Visão Geral

Nos capítulos anteriores, o arcabouço formal de RL considerava que o ambiente é totalmente observável, possui um único agente e conta com uma função de recompensa escalar explicitamente definida. O Capítulo 10 formaliza abordagens para lidar com observabilidade parcial, transferência de conhecimento entre tarefas, ausência de recompensas explícitas e interação entre múltiplos agentes [1, 2].

---

## 10.1 Observabilidade Parcial (POMDPs) e Distribuição de MDPs (Meta-RL)

### 📖 Conceito

- **POMDPs**: Quando o agente não recebe o estado real `s_t`, mas apenas uma observação parcial ou ruidosa `ω_t`, a propriedade de Markov deixa de valer na observação isolada [1, 2]. O agente precisa memorizar o histórico `h_t = (ω_0, a_0, ..., ω_t)` usando **Redes Neurais Recorrentes** (como LSTMs na arquitetura DRQN), onde o estado oculto da rede atua como um _estado de crença_ (_belief state_).

- **Meta-RL**: O agente é treinado sobre uma distribuição de MDPs `𝓜 ~ D` [1, 2]. O estado interno da RNN evolui ao longo de múltiplos episódios para aprender a se adaptar rapidamente a novas tarefas (_few-shot RL_).

### 💻 Código: DRQN (Deep Recurrent Q-Network) para POMDPs

```python
import torch
import torch.nn as nn


class DRQNNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()

        # 1. Extrator de características da observação parcial
        # do passo atual
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. Camada Recorrente LSTM:
        # Mantém o histórico h_t (Estado de Crença)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # 3. Cabeça de saída dos Q-valores
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, hidden_state):
        # Processa a observação parcial atual (ω_t)
        features = self.feature_extractor(obs)

        # Atualiza a memória (h_t, c_t)
        # incorporando a nova observação ao histórico
        h_next, c_next = self.lstm(features, hidden_state)

        # Calcula os Q-valores baseados na observação atual
        # + memória do passado
        q_values = self.q_head(h_next)

        return q_values, (h_next, c_next)
```

---

## 10.2 Aprendizado por Transferência (_Transfer Learning_)

### 📖 Conceito

O **Aprendizado por Transferência** estuda como reutilizar o conhecimento adquirido em uma tarefa de origem (_source task_) para acelerar o aprendizado em uma tarefa de destino (_target task_) [1, 2].

- **Ajuste Fino (_Fine-Tuning_)**: Congela as camadas extratoras de características e reajusta as camadas finais de decisão.

- **Redes Progressivas (_Progressive Networks_)**: Adicionam novas colunas neurais para novas tarefas enquanto mantêm os pesos das colunas antigas congelados, prevenindo o **esquecimento catastrófico** (_catastrophic forgetting_).

- **Sim-to-Real e Randomização de Domínio**: Aplica variações de física, atrito e iluminação durante simulações para criar representações robustas ao mundo real.

### 💻 Código: Redes Progressivas (Evitando Esquecimento Catastrófico)

```python
class ProgressiveColumn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, source_column=None):
        super().__init__()

        self.source_column = source_column
        # Coluna pré-treinada da Tarefa A

        if self.source_column is not None:
            # Congela os parâmetros da tarefa antiga
            # para evitar esquecimento catastrófico
            for param in self.source_column.parameters():
                param.requires_grad = False

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Conexão lateral adaptativa que recebe
        # features da coluna antiga
        self.lateral_adapter = (
            nn.Linear(hidden_dim, hidden_dim)
            if source_column
            else None
        )

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))

        # Se existir uma coluna pré-treinada,
        # combina a informação via conexão lateral
        if self.source_column is not None:
            with torch.no_grad():
                prev_h1 = torch.relu(self.source_column.fc1(x))

            h1 = h1 + torch.relu(
                self.lateral_adapter(prev_h1)
            )

        return self.fc2(h1)
```

---

## 10.3 Aprendizado Sem Função de Recompensa Explícita

### 📖 Conceito

Em problemas complexos onde definir uma função de recompensa manual é difícil ou propenso a falhas (_reward hacking_), o agente aprende a partir de demonstrações de um especialista [1, 2]:

- **Clonagem de Comportamento**: Aprendizado supervisionado direto sobre ações do especialista (sofre com a acumulação de erros compostos ao sair da distribuição de treino).

- **Aprendizado por Reforço Inverso (IRL) / GAIL**: O algoritmo infere a função de recompensa implícita. No **GAIL (_Generative Adversarial Imitation Learning_)**, um **Discriminador** aprende a diferenciar se a transição veio do especialista ou do agente, gerando um sinal de recompensa intrínseco para o agente.

### 💻 Código: Discriminador GAIL (Aprendizado por Imitação Adversário)

```python
class GAILDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
            # Saída: Probabilidade do par (s,a)
            # vir do Especialista
        )

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=-1)

        return self.net(inputs)

    def compute_intrinsic_reward(self, state, action):
        """Calcula a recompensa intrínseca para o agente de RL"""

        with torch.no_grad():
            prob_expert = self.forward(state, action)

            # Recompensa GAIL:
            # incentivado a enganar o discriminador
            # agindo como especialista
            reward = -torch.log(
                1.0 - prob_expert + 1e-8
            )

        return reward
```

---

## 10.4 Sistemas Multiagente (_Multi-Agent RL - MARL_)

### 📖 Conceito

Em **sistemas multiagente**, múltiplos agentes interagem no mesmo ambiente de forma cooperativa, competitiva ou mista [1, 2].

- **Não-Estacionariedade**: Como todos os agentes mudam suas políticas simultaneamente durante o treino, o ambiente deixa de ser estacionário do ponto de vista de um único agente.

- **CTDE (_Centralized Training, Decentralized Execution_)**: Durante o treinamento, um **Crítico Centralizado** enxerga os estados e ações de todos os agentes para estabilizar a estimativa de valor. Na execução, cada **Ator Descentralizado** toma decisões baseado unicamente na sua observação local.

### 💻 Código: Arquitetura CTDE (Ator Descentralizado e Crítico Centralizado)

```python
class CentralizedCritic(nn.Module):
    """CRÍTICO CENTRALIZADO: Utilizado exclusivamente durante o TREINAMENTO"""

    def __init__(self, global_state_dim, all_actions_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(global_state_dim + all_actions_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            # Avalia o valor do estado global do jogo
        )

    def forward(self, global_state, all_actions):
        inputs = torch.cat(
            [global_state, all_actions],
            dim=-1
        )

        return self.net(inputs)


class DecentralizedActor(nn.Module):
    """ATOR DESCENTRALIZADO: Utilizado por cada agente na EXECUÇÃO/TESTE"""

    def __init__(self, local_obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
            # Distribuição de ações baseada APENAS na visão local
        )

    def forward(self, local_obs):
        return self.net(local_obs)
```

---

# 🧠 Resumo do Capítulo

O Capítulo 10 amplia o aprendizado por reforço profundo para cenários em que as hipóteses tradicionais dos MDPs não são suficientes:

| Problema                      | Abordagem                       | Ideia principal                                             |
| ----------------------------- | ------------------------------- | ----------------------------------------------------------- |
| Observação parcial            | POMDP + DRQN                    | Usar memória para inferir informações ocultas               |
| Adaptação entre tarefas       | Meta-RL                         | Aprender a aprender novas tarefas rapidamente               |
| Reutilização de conhecimento  | Transfer Learning               | Aproveitar conhecimento de tarefas anteriores               |
| Esquecimento catastrófico     | Progressive Networks            | Manter conhecimento antigo enquanto aprende uma nova tarefa |
| Ausência de recompensa manual | Imitation Learning / IRL / GAIL | Aprender a partir de demonstrações                          |
| Múltiplos agentes             | MARL                            | Aprender em ambientes com vários agentes                    |
| Instabilidade em MARL         | CTDE                            | Treinamento centralizado + execução descentralizada         |

---

# 🎯 Ideia Central

O RL tradicional assume um MDP relativamente bem definido:

```text
Estado → Ação → Recompensa → Próximo Estado
```

O Capítulo 10 mostra como lidar com situações mais próximas de problemas reais:

### Observação parcial

```text
Observação parcial
        ↓
Memória / RNN
        ↓
Decisão
```

### Múltiplas tarefas

```text
Múltiplas tarefas
        ↓
Meta-RL / Transfer Learning
        ↓
Adaptação
```

### Sem recompensa explícita

```text
Demonstrações
        ↓
Imitation Learning / GAIL
        ↓
Aprendizado
```

### Múltiplos agentes

```text
MARL
        ↓
CTDE
        ↓
Treinamento centralizado
        +
Execução descentralizada
```

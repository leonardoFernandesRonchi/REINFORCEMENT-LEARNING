# 📚 Estudos de Aprendizado por Reforço (Deep RL)

## 📌 Índice

### 1. Definições e Conceitos Fundamentais

- [1.1. Propriedade de Markov e MDPs](#11-propriedade-de-markov-e-mdps)
- [1.2. Q-Learning Tabular](#12-q-learning-tabular)
- [1.3. Fitted Q-Learning & Deep Q-Networks (DQN)](#13-fitted-q-learning--deep-q-networks-dqn)
- [1.4. Métodos de Gradiente de Política e Actor-Critic](#14-métodos-de-gradiente-de-política-e-actor-critic)

- [1.5. Double DQN](#15-double-dqn)
- [1.6. Dueling Network Architecture](#16-dueling-network-architecture)
- [1.7. Distributional DQN](#17-distributional-dqn)
- [1.8. Multi-step Learning](#18-multi-step-learning)
- [1.9. Combinação das melhorias e variantes do DQN](#19-combinacao-das-melhorias-e-variantes-do-dqn)

#### 1.5. Double DQN

O **Double DQN** é uma melhoria do DQN tradicional que reduz o problema da **superestimação dos valores Q**. No DQN convencional, a mesma rede utilizada para escolher a melhor ação também é usada para avaliar essa ação:

```text
maxₐ' Q(s', a'; θ_target)
```

Esse procedimento pode favorecer valores excessivamente altos, principalmente quando as estimativas da rede ainda são imprecisas.

No Double DQN, a seleção da melhor ação é feita pela **rede principal**, enquanto a avaliação dessa ação é feita pela **rede alvo**:

```text
a* = argmaxₐ' Q(s', a'; θ)
Y = r + γ Q(s', a*; θ_target)
```

Assim, as duas redes desempenham funções diferentes:

- **Rede principal (`θ`)** — escolhe qual ação parece ser a melhor.
- **Rede alvo (`θ_target`)** — estima o valor da ação escolhida.

Essa separação reduz a tendência de superestimar as recompensas futuras e pode tornar o treinamento mais estável.

**Quando usar:** quando o DQN apresenta valores Q exageradamente altos, instabilidade ou dificuldade para distinguir ações de qualidade semelhante.

**Quando não usar:** em ambientes muito simples, nos quais o DQN tradicional já apresenta desempenho satisfatório e a complexidade adicional não traz benefício significativo.

[💻 Ver exemplo de código de Double DQN](#código-5-double-dqn)

---

## 1.6. Dueling Network Architecture

A **Dueling Network Architecture** modifica a estrutura interna da rede neural do DQN. Em vez de produzir diretamente um Q-value para cada ação, a rede é dividida em dois fluxos:

1. **Value stream** — estima o valor geral do estado `V(s)`, independentemente da ação.
2. **Advantage stream** — estima a vantagem de cada ação `A(s, a)` em relação às outras ações disponíveis.

Esses dois fluxos são combinados para produzir os valores Q:

```text
Q(s, a) = V(s) + A(s, a)
```

Na prática, é comum utilizar uma normalização da vantagem para evitar ambiguidades entre `V(s)` e `A(s, a)`:

```text
Q(s, a) = V(s) + A(s, a) - meanₐ'[A(s, a')]
```

A principal vantagem dessa arquitetura é que ela consegue aprender **quão bom é um estado** mesmo quando várias ações possuem efeitos semelhantes. Isso pode ser útil em situações nas quais a escolha exata da ação não é tão importante.

**Quando usar:** em ambientes com muitas ações semelhantes ou nos quais é importante identificar rapidamente se um estado é bom ou ruim.

**Quando não usar:** quando a rede já é muito pequena ou quando a divisão em dois fluxos aumenta a complexidade sem produzir melhorias mensuráveis.

[💻 Ver exemplo de código de Dueling DQN](#código-6-dueling-dqn)

---

## 1.7. Distributional DQN

O **Distributional DQN** não estima apenas o valor esperado de uma ação. Ele procura aprender a **distribuição completa dos retornos futuros** associados a essa ação.

No DQN tradicional, a rede estima:

```text
Q(s, a) = E[Z(s, a)]
```

Nesse caso, `Q(s, a)` representa somente o retorno esperado.

No Distributional DQN, a variável aleatória `Z(s, a)` representa diferentes retornos possíveis. A rede tenta modelar essa distribuição, permitindo representar situações como:

- uma ação com retorno médio alto, mas muito arriscada;
- uma ação com retorno médio semelhante, porém mais consistente;
- uma ação com possibilidade de recompensa muito alta ou muito baixa.

A distribuição pode ser representada de diferentes maneiras. Uma abordagem conhecida é o **Categorical DQN (C51)**, que utiliza um conjunto fixo de valores possíveis, chamados de _atoms_, e aprende a probabilidade associada a cada um deles.

Em vez de retornar apenas:

```text
Q(s, a) = 10
```

a rede pode representar algo semelhante a:

```text
Retorno 5  → probabilidade 0,3
Retorno 10 → probabilidade 0,5
Retorno 20 → probabilidade 0,2
```

O objetivo é aprender como os retornos estão distribuídos, e não somente sua média.

**Quando usar:** em ambientes com incerteza elevada, recompensas muito variáveis ou nos quais a distribuição dos resultados fornece informações importantes.

**Quando não usar:** quando o problema é simples e o retorno esperado já é suficiente para escolher boas ações.

[💻 Ver exemplo de código de Distributional DQN (C51)](#código-7-distributional-dqn-c51)

---

## 1.8. Multi-step Learning

O **Multi-step Learning** utiliza informações de várias transições futuras para atualizar o valor de uma ação. O DQN tradicional normalmente utiliza um retorno de um único passo:

```text
Y₁ = rₜ + γ maxₐ Q(sₜ₊₁, a)
```

No aprendizado de `n` passos, o alvo considera várias recompensas consecutivas:

```text
Yₙ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... + γⁿ maxₐ Q(sₜ₊ₙ, a)
```

Por exemplo, em um método de três passos, o agente considera:

1. a recompensa recebida no instante atual;
2. a recompensa recebida no passo seguinte;
3. a recompensa recebida no segundo passo seguinte;
4. a estimativa futura a partir do terceiro estado.

Essa abordagem faz com que as recompensas sejam propagadas mais rapidamente para as ações que contribuíram para obtê-las. Porém, utilizar muitos passos pode aumentar a variância das atualizações e tornar o aprendizado mais sensível às estimativas incorretas.

**Quando usar:** quando as recompensas são atrasadas e é necessário propagar informações de longo alcance mais rapidamente.

**Quando não usar:** quando o ambiente possui muito ruído ou quando o uso de muitos passos provoca atualizações instáveis.

[💻 Ver exemplo de código de Multi-step Learning](#código-8-multi-step-learning)

---

## 1.9. Combinação das melhorias e variantes do DQN

As melhorias do DQN podem ser combinadas para aproveitar diferentes vantagens. Uma combinação conhecida é o **Rainbow DQN**, que reúne várias técnicas em um único agente.

Entre as principais melhorias utilizadas estão:

- **Double DQN** — reduz a superestimação dos valores Q.
- **Dueling Network Architecture** — separa o valor do estado da vantagem de cada ação.
- **Distributional DQN** — aprende a distribuição dos retornos.
- **Multi-step Learning** — utiliza recompensas de vários passos.
- **Prioritized Experience Replay** — prioriza experiências consideradas mais relevantes para o aprendizado.
- **Noisy Networks** — utiliza ruído parametrizado para favorecer a exploração.

A combinação dessas técnicas pode produzir um agente mais eficiente e robusto do que o DQN básico. Entretanto, o treinamento também se torna mais complexo, pois há mais hiperparâmetros, componentes e possibilidades de instabilidade.

### Principais variantes do DQN

Além do DQN original, existem diversas variantes:

| Variante                          | Principal característica                                   |
| --------------------------------- | ---------------------------------------------------------- |
| **Double DQN**                    | Reduz a superestimação dos valores Q.                      |
| **Dueling DQN**                   | Separa o valor do estado e a vantagem das ações.           |
| **Distributional DQN**            | Aprende a distribuição dos retornos possíveis.             |
| **Multi-step DQN**                | Utiliza recompensas acumuladas de vários passos.           |
| **Prioritized Experience Replay** | Amostra com maior frequência experiências mais relevantes. |
| **Noisy DQN**                     | Introduz ruído aprendido nos parâmetros para exploração.   |
| **Rainbow DQN**                   | Combina várias melhorias do DQN em um único agente.        |

**Quando usar uma combinação:** quando o ambiente é complexo, possui recompensas atrasadas, elevada incerteza ou exige melhor eficiência de amostragem.

**Quando não usar uma combinação:** durante os primeiros estudos ou em ambientes simples, nos quais é mais adequado começar com DQN básico e adicionar uma melhoria por vez. Dessa forma, fica mais fácil compreender o efeito de cada técnica e identificar a origem de eventuais problemas.

[💻 Ver exemplo de código da combinação das melhorias](#código-9-combinação-das-melhorias--mini-rainbow)

---

# 2. Exemplos de Código Prático

- [💻 Código 1: Simulação com a Propriedade de Markov](#código-1-simulação-com-a-propriedade-de-markov)
- [💻 Código 2: Algoritmo Q-Learning Tabular](#código-2-algoritmo-q-learning-tabular)
- [💻 Código 3: Agente Deep Q-Network (DQN)](#código-3-agente-deep-q-network-dqn)
- [💻 Código 4: Algoritmo REINFORCE (Policy Gradient)](#código-4-algoritmo-reinforce-policy-gradient)

- [💻 Código 5: Double DQN](#código-5-double-dqn)
- [💻 Código 6: Dueling DQN](#código-6-dueling-dqn)
- [💻 Código 7: Distributional DQN (C51)](#código-7-distributional-dqn-c51)
- [💻 Código 8: Multi-step Learning](#código-8-multi-step-learning)
- [💻 Código 9: Combinação das melhorias — Mini Rainbow](#código-9-combinação-das-melhorias--mini-rainbow)

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

## 1.5. Double DQN

O **Double DQN** é uma melhoria do DQN tradicional que reduz o problema da **superestimação dos valores Q**. No DQN convencional, a mesma rede utilizada para escolher a melhor ação também é usada para avaliar essa ação:

```text
maxₐ' Q(s', a'; θ_target)
```

Esse procedimento pode favorecer valores excessivamente altos, principalmente quando as estimativas da rede ainda são imprecisas.

No Double DQN, a seleção da melhor ação é feita pela **rede principal**, enquanto a avaliação dessa ação é feita pela **rede alvo**:

```text
a* = argmaxₐ' Q(s', a'; θ)
Y = r + γ Q(s', a*; θ_target)
```

Assim, as duas redes desempenham funções diferentes:

- **Rede principal (`θ`)** — escolhe qual ação parece ser a melhor.
- **Rede alvo (`θ_target`)** — estima o valor da ação escolhida.

Essa separação reduz a tendência de superestimar as recompensas futuras e pode tornar o treinamento mais estável.

**Quando usar:** quando o DQN apresenta valores Q exageradamente altos, instabilidade ou dificuldade para distinguir ações de qualidade semelhante.

**Quando não usar:** em ambientes muito simples, nos quais o DQN tradicional já apresenta desempenho satisfatório e a complexidade adicional não traz benefício significativo.

---

## 1.6. Dueling Network Architecture

A **Dueling Network Architecture** modifica a estrutura interna da rede neural do DQN. Em vez de produzir diretamente um Q-value para cada ação, a rede é dividida em dois fluxos:

1. **Value stream** — estima o valor geral do estado `V(s)`, independentemente da ação.
2. **Advantage stream** — estima a vantagem de cada ação `A(s, a)` em relação às outras ações disponíveis.

Esses dois fluxos são combinados para produzir os valores Q:

```text
Q(s, a) = V(s) + A(s, a)
```

Na prática, é comum utilizar uma normalização da vantagem para evitar ambiguidades entre `V(s)` e `A(s, a)`:

```text
Q(s, a) = V(s) + A(s, a) - meanₐ'[A(s, a')]
```

A principal vantagem dessa arquitetura é que ela consegue aprender **quão bom é um estado** mesmo quando várias ações possuem efeitos semelhantes. Isso pode ser útil em situações nas quais a escolha exata da ação não é tão importante.

**Quando usar:** em ambientes com muitas ações semelhantes ou nos quais é importante identificar rapidamente se um estado é bom ou ruim.

**Quando não usar:** quando a rede já é muito pequena ou quando a divisão em dois fluxos aumenta a complexidade sem produzir melhorias mensuráveis.

---

## 1.7. Distributional DQN

O **Distributional DQN** não estima apenas o valor esperado de uma ação. Ele procura aprender a **distribuição completa dos retornos futuros** associados a essa ação.

No DQN tradicional, a rede estima:

```text
Q(s, a) = E[Z(s, a)]
```

Nesse caso, `Q(s, a)` representa somente o retorno esperado.

No Distributional DQN, a variável aleatória `Z(s, a)` representa diferentes retornos possíveis. A rede tenta modelar essa distribuição, permitindo representar situações como:

- uma ação com retorno médio alto, mas muito arriscada;
- uma ação com retorno médio semelhante, porém mais consistente;
- uma ação com possibilidade de recompensa muito alta ou muito baixa.

A distribuição pode ser representada de diferentes maneiras. Uma abordagem conhecida é o **Categorical DQN (C51)**, que utiliza um conjunto fixo de valores possíveis, chamados de _atoms_, e aprende a probabilidade associada a cada um deles.

Em vez de retornar apenas:

```text
Q(s, a) = 10
```

a rede pode representar algo semelhante a:

```text
Retorno 5  → probabilidade 0,3
Retorno 10 → probabilidade 0,5
Retorno 20 → probabilidade 0,2
```

O objetivo é aprender como os retornos estão distribuídos, e não somente sua média.

**Quando usar:** em ambientes com incerteza elevada, recompensas muito variáveis ou nos quais a distribuição dos resultados fornece informações importantes.

**Quando não usar:** quando o problema é simples e o retorno esperado já é suficiente para escolher boas ações.

---

## 1.8. Multi-step Learning

O **Multi-step Learning** utiliza informações de várias transições futuras para atualizar o valor de uma ação. O DQN tradicional normalmente utiliza um retorno de um único passo:

```text
Y₁ = rₜ + γ maxₐ Q(sₜ₊₁, a)
```

No aprendizado de `n` passos, o alvo considera várias recompensas consecutivas:

```text
Yₙ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... + γⁿ maxₐ Q(sₜ₊ₙ, a)
```

Por exemplo, em um método de três passos, o agente considera:

1. a recompensa recebida no instante atual;
2. a recompensa recebida no passo seguinte;
3. a recompensa recebida no segundo passo seguinte;
4. a estimativa futura a partir do terceiro estado.

Essa abordagem faz com que as recompensas sejam propagadas mais rapidamente para as ações que contribuíram para obtê-las. Porém, utilizar muitos passos pode aumentar a variância das atualizações e tornar o aprendizado mais sensível às estimativas incorretas.

**Quando usar:** quando as recompensas são atrasadas e é necessário propagar informações de longo alcance mais rapidamente.

**Quando não usar:** quando o ambiente possui muito ruído ou quando o uso de muitos passos provoca atualizações instáveis.

---

## 1.9. Combinação das melhorias e variantes do DQN

As melhorias do DQN podem ser combinadas para aproveitar diferentes vantagens. Uma combinação conhecida é o **Rainbow DQN**, que reúne várias técnicas em um único agente.

Entre as principais melhorias utilizadas estão:

- **Double DQN** — reduz a superestimação dos valores Q.
- **Dueling Network Architecture** — separa o valor do estado da vantagem de cada ação.
- **Distributional DQN** — aprende a distribuição dos retornos.
- **Multi-step Learning** — utiliza recompensas de vários passos.
- **Prioritized Experience Replay** — prioriza experiências consideradas mais relevantes para o aprendizado.
- **Noisy Networks** — utiliza ruído parametrizado para favorecer a exploração.

A combinação dessas técnicas pode produzir um agente mais eficiente e robusto do que o DQN básico. Entretanto, o treinamento também se torna mais complexo, pois há mais hiperparâmetros, componentes e possibilidades de instabilidade.

### Principais variantes do DQN

Além do DQN original, existem diversas variantes:

| Variante                          | Principal característica                                   |
| --------------------------------- | ---------------------------------------------------------- |
| **Double DQN**                    | Reduz a superestimação dos valores Q.                      |
| **Dueling DQN**                   | Separa o valor do estado e a vantagem das ações.           |
| **Distributional DQN**            | Aprende a distribuição dos retornos possíveis.             |
| **Multi-step DQN**                | Utiliza recompensas acumuladas de vários passos.           |
| **Prioritized Experience Replay** | Amostra com maior frequência experiências mais relevantes. |
| **Noisy DQN**                     | Introduz ruído aprendido nos parâmetros para exploração.   |
| **Rainbow DQN**                   | Combina várias melhorias do DQN em um único agente.        |

**Quando usar uma combinação:** quando o ambiente é complexo, possui recompensas atrasadas, elevada incerteza ou exige melhor eficiência de amostragem.

**Quando não usar uma combinação:** durante os primeiros estudos ou em ambientes simples, nos quais é mais adequado começar com DQN básico e adicionar uma melhoria por vez. Dessa forma, fica mais fácil compreender o efeito de cada técnica e identificar a origem de eventuais problemas.

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


# ============================================================
# 1. CONFIGURAÇÃO DO AMBIENTE
# ============================================================

# Cria o ambiente "FrozenLake" (Lago Congelado).
#
# O mapa padrão possui 4x4 posições:
#
#     0   1   2   3
#     4   5   6   7
#     8   9  10  11
#    12  13  14  15
#
# Portanto, temos 16 estados possíveis.
#
# is_slippery=False significa que o gelo NÃO escorrega.
# Se o agente escolher "direita", ele realmente irá para a direita.
#
# Se fosse True, poderia acontecer de o agente tentar ir
# para a direita e acabar indo para outra direção.

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)


# ============================================================
# 2. INICIALIZAÇÃO DA TABELA Q
# ============================================================

# A Q-Table é o "cérebro" do agente.
#
# Ela guarda:
#
#     "Quanto vale tomar determinada ação em determinado estado?"
#
# Temos:
#
#     16 estados
#     4 ações
#
# Portanto, nossa tabela terá:
#
#     16 linhas x 4 colunas
#
# Cada LINHA representa um estado.
#
# Cada COLUNA representa uma ação.
#
# A tabela inicialmente é preenchida com zeros porque
# o agente ainda não aprendeu nada.

q_table = np.zeros((
    env.observation_space.n,   # 16 estados (linhas)
    env.action_space.n          # 4 ações (colunas)
))


# A tabela começa aproximadamente assim:
#
#              ←    →    ↑    ↓
#
# estado 0     0    0    0    0
# estado 1     0    0    0    0
# estado 2     0    0    0    0
# estado 3     0    0    0    0
# ...
# estado 15    0    0    0    0
#
# Ou seja:
#
# q_table[estado, ação]
#
# representa o valor de determinada ação em determinado estado.


# ============================================================
# 3. HIPERPARÂMETROS
# ============================================================

# ALPHA = taxa de aprendizado.
#
# Define quanto da nova informação será incorporada
# na informação que já estava na Q-Table.
#
# 0.1 = 10%
#
# Portanto, o agente não substitui completamente o valor antigo.
# Ele se aproxima 10% do novo valor descoberto.

alpha = 0.1


# GAMMA = fator de desconto.
#
# Define o quanto o agente valoriza recompensas futuras.
#
# 0.99 significa que recompensas futuras são consideradas
# muito importantes.

gamma = 0.99


# EPSILON = taxa de exploração.
#
# 10% das vezes o agente escolherá uma ação aleatória
# para tentar descobrir coisas novas.
#
# Nos outros 90%, ele escolherá a ação que atualmente
# considera melhor.

epsilon = 0.1


# ============================================================
# 4. LOOP DE TREINAMENTO
# ============================================================

# O agente jogará 1000 partidas (episódios).

for episode in range(1000):


    # --------------------------------------------------------
    # COMEÇO DE UM NOVO EPISÓDIO
    # --------------------------------------------------------

    # Reinicia o FrozenLake.
    #
    # state = estado inicial do agente.
    #
    # O "_" recebe informações extras que não vamos utilizar.

    state, _ = env.reset()


    # Controla se o episódio terminou.
    #
    # False = ainda está jogando
    # True  = episódio terminou

    done = False


    # Continua jogando enquanto o episódio não terminar.

    while not done:


        # ====================================================
        # 5. ESCOLHA DA AÇÃO
        # ====================================================
        #
        # Utilizamos a estratégia EPSILON-GREEDY.
        #
        # O agente possui duas possibilidades:
        #
        # 1. EXPLORAR
        #    Escolher uma ação aleatória.
        #
        # 2. EXPLOTAR
        #    Escolher a melhor ação que ele conhece atualmente.
        #
        # Como epsilon = 0.1:
        #
        # 10% -> exploração
        # 90% -> exploração do conhecimento atual


        # Sorteia um número entre 0 e 1.

        if np.random.uniform(0, 1) < epsilon:


            # ------------------------------------------------
            # EXPLORAÇÃO
            # ------------------------------------------------
            #
            # O agente ignora temporariamente o que aprendeu
            # e escolhe uma ação aleatória.
            #
            # Isso permite descobrir caminhos novos.

            action = env.action_space.sample()


        else:


            # ------------------------------------------------
            # EXPLOTAÇÃO
            # ------------------------------------------------
            #
            # Agora o agente olha para a linha correspondente
            # ao estado atual.
            #
            # Por exemplo, se:
            #
            # state = 0
            #
            # então:
            #
            # q_table[0]
            #
            # pode ser:
            #
            # [0.2, 0.8, 0.1, 0.4]
            #
            # O maior valor é 0.8.
            #
            # np.argmax() retorna a posição desse maior valor.
            #
            # Nesse exemplo:
            #
            # np.argmax([0.2, 0.8, 0.1, 0.4])
            #
            # retorna:
            #
            # 1
            #
            # Portanto, a ação 1 será escolhida.

            action = np.argmax(q_table[state])


        # ====================================================
        # 6. INTERAÇÃO COM O AMBIENTE
        # ====================================================

        # O agente executa a ação que escolheu.
        #
        # O ambiente então responde dizendo o que aconteceu.
        #
        # Recebemos:
        #
        # next_state -> próximo estado
        # reward     -> recompensa recebida
        # terminated -> terminou naturalmente?
        # truncated  -> terminou por limite de passos?
        # _          -> informação extra que não utilizaremos

        next_state, reward, terminated, truncated, _ = env.step(action)


        # O episódio terminou se:
        #
        # - o agente chegou ao objetivo;
        # - o agente caiu em um buraco;
        # - ou atingiu o limite de passos.
        #
        # Por isso usamos:
        #
        # terminated OR truncated

        done = terminated or truncated


        # ====================================================
        # 7. ATUALIZAÇÃO DA Q-TABLE
        # ====================================================

        # Aqui está a parte mais importante do Q-Learning.
        #
        # O agente acabou de fazer:
        #
        #     estado atual
        #          ↓
        #       ação
        #          ↓
        #     próximo estado
        #
        # Por exemplo:
        #
        #     estado 0
        #        ↓
        #      ação
        #        ↓
        #     estado 4
        #
        # Agora precisamos perguntar:
        #
        # "O estado 4 parece ser um estado bom ou ruim?"


        # ----------------------------------------------------
        # 7.1 ENCONTRAR A MELHOR AÇÃO NO PRÓXIMO ESTADO
        # ----------------------------------------------------

        # Primeiro pegamos a linha do próximo estado.
        #
        # Se:
        #
        # next_state = 4
        #
        # então:
        #
        # q_table[next_state]
        #
        # é:
        #
        # q_table[4]
        #
        # Imagine que seja:
        #
        # [0.0, 0.0, 0.8, 0.3]
        #
        # Isso significa:
        #
        # ação 0 -> valor 0.0
        # ação 1 -> valor 0.0
        # ação 2 -> valor 0.8
        # ação 3 -> valor 0.3
        #
        # A melhor ação seria a ação 2.


        best_next_action = np.argmax(
            q_table[next_state]
        )


        # ----------------------------------------------------
        # 7.2 CALCULAR O TARGET
        # ----------------------------------------------------

        # Agora calculamos quanto a ação anterior
        # deveria valer.
        #
        # A fórmula é:
        #
        #     TARGET =
        #
        #     recompensa recebida AGORA
        #
        #     +
        #
        #     recompensa futura estimada
        #
        # A recompensa futura é multiplicada por GAMMA.
        #
        # Fórmula:
        #
        #     reward + gamma * melhor_valor_do_próximo_estado
        #
        #
        # Imagine:
        #
        # reward = 0
        #
        # gamma = 0.99
        #
        # e descobrimos que:
        #
        # q_table[next_state, best_next_action] = 0.8
        #
        # Então:
        #
        #     target = 0 + 0.99 * 0.8
        #
        #     target = 0.792
        #
        # Isso significa:
        #
        # "Eu não ganhei nada AGORA,
        # mas o próximo estado parece ser muito bom.
        # Portanto, a ação que me trouxe até aqui
        # também deve ter algum valor."


        td_target = (
            reward
            + gamma * q_table[next_state, best_next_action]
        )


        # ----------------------------------------------------
        # 7.3 ATUALIZAR O VALOR NA Q-TABLE
        # ----------------------------------------------------

        # Agora atualizamos o valor da ação que o agente
        # ACABOU DE TOMAR.
        #
        # É importante perceber isso:
        #
        # q_table[state, action]
        #
        # representa:
        #
        # "Quanto eu achava que valia tomar essa ação
        # naquele estado?"
        #
        #
        # Imagine:
        #
        # state = 0
        # action = 1
        #
        # Então estamos atualizando:
        #
        # q_table[0, 1]


        # A fórmula é:
        #
        # NOVO VALOR =
        #
        # VALOR ANTIGO
        #
        # +
        #
        # ALPHA *
        #
        # (TARGET - VALOR ANTIGO)
        #
        #
        # Imagine:
        #
        # valor antigo = 0
        # target       = 0.792
        # alpha        = 0.1
        #
        #
        # Então:
        #
        #     0 + 0.1 * (0.792 - 0)
        #
        #     0 + 0.0792
        #
        #     = 0.0792
        #
        #
        # Portanto, a Q-Table passa de:
        #
        # [0, 0, 0, 0]
        #
        # para algo como:
        #
        # [0, 0.0792, 0, 0]
        #
        #
        # O agente acabou de aprender que aquela ação
        # parece ter algum valor.


        q_table[state, action] += (
            alpha
            * (td_target - q_table[state, action])
        )


        # ====================================================
        # 8. AVANÇAR PARA O PRÓXIMO ESTADO
        # ====================================================

        # Agora o agente realmente "avança" no ambiente.
        #
        # Antes:
        #
        #     state = 0
        #
        # Ele executou uma ação e chegou ao:
        #
        #     next_state = 4
        #
        # Então fazemos:
        #
        #     state = next_state
        #
        # Agora:
        #
        #     state = 4
        #
        # Na próxima repetição do while, o agente estará
        # tomando uma decisão a partir do estado 4.

        state = next_state


# ============================================================
# 9. FINALIZAÇÃO
# ============================================================

# Depois dos 1000 episódios, encerramos o ambiente
# para liberar os recursos utilizados.

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


# ============================================================
# 1. REDE NEURAL
# ============================================================

# Esta classe representa o "cérebro" do nosso agente.
#
# Entrada:
#   estado do ambiente
#
# Saída:
#   um Q-value para cada ação possível
#
# No CartPole existem 2 ações:
#   0 = mover para a esquerda
#   1 = mover para a direita
#
# Portanto, a rede recebe 4 números e devolve 2 números.
#
# Exemplo:
#
# Estado:
# [0.1, -0.2, 0.05, 0.3]
#
# Rede:
# [0.72, 0.35]
#
# Isso significa:
#
# Q(estado, esquerda)  = 0.72
# Q(estado, direita)   = 0.35
#
# Portanto, neste estado, a rede considera
# "esquerda" a melhor ação.


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):

        # Inicializa a classe pai nn.Module.
        # Isso é necessário para o PyTorch reconhecer
        # esta classe como uma rede neural.
        super().__init__()

        # Cria a arquitetura da rede.
        self.fc = nn.Sequential(

            # Primeira camada:
            #
            # Recebe state_dim números.
            # No CartPole:
            # state_dim = 4
            #
            # Produz 64 números.
            nn.Linear(state_dim, 64),

            # Função de ativação.
            #
            # Permite que a rede aprenda relações
            # não lineares entre os dados.
            nn.ReLU(),

            # Última camada:
            #
            # Recebe os 64 valores anteriores
            # e produz action_dim valores.
            #
            # No CartPole:
            # action_dim = 2
            #
            # Portanto:
            #
            # 4 entradas -> 64 -> 2 saídas
            nn.Linear(64, action_dim)
        )

    def forward(self, x):

        # Quando fazemos:
        #
        # q_net(x)
        #
        # o PyTorch chama automaticamente este método.
        #
        # Aqui simplesmente passamos o estado
        # pela rede neural.
        return self.fc(x)


# ============================================================
# 2. CRIANDO O AMBIENTE
# ============================================================

# Cria o jogo CartPole.
#
# O objetivo é manter o bastão equilibrado
# em cima do carrinho pelo maior tempo possível.
env = gym.make("CartPole-v1")


# Quantos números existem no estado?
#
# CartPole possui 4 informações:
#
# 1. posição do carrinho
# 2. velocidade do carrinho
# 3. ângulo do bastão
# 4. velocidade angular do bastão
state_dim = env.observation_space.shape[0]


# Quantas ações existem?
#
# CartPole possui:
#
# 0 = esquerda
# 1 = direita
action_dim = env.action_space.n


# ============================================================
# 3. CRIANDO AS DUAS REDES DO DQN
# ============================================================

# Rede Principal.
#
# É a rede que:
# - escolhe ações
# - recebe treinamento
# - tem seus pesos modificados pelo optimizer
q_net = QNetwork(state_dim, action_dim)


# Rede Alvo.
#
# Ela serve como uma referência mais estável
# para calcular os valores que a rede principal
# deveria aprender.
target_net = QNetwork(state_dim, action_dim)


# Inicialmente copiamos os pesos da rede principal
# para a rede alvo.
#
# Portanto:
#
# q_net = conhecimento inicial
# target_net = exatamente o mesmo conhecimento
target_net.load_state_dict(q_net.state_dict())


# Adam é o algoritmo responsável por ajustar
# os pesos da rede neural durante o aprendizado.
#
# lr = learning rate.
#
# 0.001 significa o tamanho dos passos usados
# para modificar os pesos.
optimizer = optim.Adam(
    q_net.parameters(),
    lr=0.001
)


# ============================================================
# 4. MEMÓRIA DE EXPERIÊNCIAS
# ============================================================

# A memória guarda as experiências do agente.
#
# Cada experiência possui:
#
# (estado, ação, recompensa, próximo_estado, terminou)
#
# Exemplo:
#
# (
#   [0.1, 0.2, -0.1, 0.3],
#   1,
#   1,
#   [0.2, 0.3, -0.05, 0.4],
#   False
# )
#
# maxlen=10000 significa que podemos guardar
# no máximo 10.000 experiências.
#
# Quando estiver cheia, experiências antigas
# são automaticamente descartadas.
replay_memory = deque(maxlen=10000)


# ============================================================
# 5. HIPERPARÂMETROS
# ============================================================

# Gamma:
#
# Define quanto o agente valoriza recompensas futuras.
#
# 0.99 = valoriza bastante o futuro.
#
# A ideia é:
#
# recompensa imediata
# +
# recompensa futura descontada
gamma = 0.99


# Quantas experiências serão utilizadas
# em cada treinamento.
#
# Em vez de aprender usando apenas uma experiência,
# pegamos 64 experiências aleatórias da memória.
batch_size = 64


# A cada quantos passos atualizamos a Rede Alvo.
#
# A cada 100 passos:
#
# target_net <- q_net
C = 100


# Epsilon controla exploração.
#
# 10% das vezes:
#   ação aleatória
#
# 90% das vezes:
#   melhor ação conhecida pela rede
epsilon = 0.1


# ============================================================
# 6. PRIMEIRO ESTADO
# ============================================================

# Reinicia o ambiente.
#
# state contém o estado inicial.
state, _ = env.reset()


# ============================================================
# 7. LOOP PRINCIPAL
# ============================================================

# Vamos executar 1000 passos.
#
# IMPORTANTE:
#
# Isso são 1000 interações com o ambiente,
# NÃO 1000 episódios.
for step in range(1, 1001):


    # --------------------------------------------------------
    # TRANSFORMAR O ESTADO EM TENSOR
    # --------------------------------------------------------

    # O ambiente devolve um array NumPy.
    #
    # O PyTorch trabalha com Tensors.
    #
    # FloatTensor transforma os números em Tensor.
    #
    # Exemplo:
    #
    # NumPy:
    # [0.1, 0.2, -0.1, 0.3]
    #
    # Tensor:
    # tensor([0.1, 0.2, -0.1, 0.3])
    state_tensor = torch.FloatTensor(state).unsqueeze(0)


    # --------------------------------------------------------
    # ESCOLHER AÇÃO
    # --------------------------------------------------------

    # Gera um número aleatório entre 0 e 1.
    #
    # Se for menor que epsilon:
    # exploração.
    #
    # Caso contrário:
    # exploração do conhecimento atual.
    if random.random() < epsilon:

        # EXPLORAÇÃO
        #
        # Escolhe uma ação aleatória.
        action = env.action_space.sample()

    else:

        # EXPLOTAÇÃO
        #
        # Não precisamos calcular gradientes aqui,
        # porque estamos apenas perguntando à rede
        # qual ação ela considera melhor.
        with torch.no_grad():

            # A rede recebe o estado.
            #
            # Exemplo:
            #
            # q_net(state)
            #
            # poderia produzir:
            #
            # [0.73, 0.42]
            #
            # torch.argmax encontra o índice
            # do maior valor.
            #
            # 0.73 é maior que 0.42
            #
            # portanto:
            #
            # action = 0
            action = torch.argmax(
                q_net(state_tensor)
            ).item()


    # --------------------------------------------------------
    # EXECUTAR A AÇÃO
    # --------------------------------------------------------

    # Envia a ação escolhida para o ambiente.
    #
    # O ambiente responde:
    #
    # next_state = próximo estado
    # reward = recompensa recebida
    # terminated = terminou naturalmente
    # truncated = terminou por limite de tempo
    next_state, reward, terminated, truncated, _ = env.step(action)


    # Se terminou por qualquer motivo:
    # done = True
    done = terminated or truncated


    # --------------------------------------------------------
    # GUARDAR EXPERIÊNCIA
    # --------------------------------------------------------

    # Guardamos tudo que acabou de acontecer.
    #
    # O agente poderá rever essa experiência
    # posteriormente durante o treinamento.
    replay_memory.append((
        state,
        action,
        reward,
        next_state,
        done
    ))


    # --------------------------------------------------------
    # ATUALIZAR O ESTADO
    # --------------------------------------------------------

    if done:

        # Se o episódio acabou,
        # começamos um novo episódio.
        state, _ = env.reset()

    else:

        # Se ainda não acabou,
        # o próximo estado vira o estado atual.
        state = next_state


    # ========================================================
    # 8. TREINAMENTO DA REDE
    # ========================================================

    # Não podemos treinar imediatamente se temos
    # poucas experiências.
    #
    # Precisamos ter pelo menos 64.
    if len(replay_memory) >= batch_size:


        # Escolhe aleatoriamente 64 experiências
        # da memória.
        #
        # Isso é o Experience Replay.
        batch = random.sample(
            replay_memory,
            batch_size
        )


        # Separa as experiências em grupos.
        #
        # Antes:
        #
        # (estado, ação, recompensa, próximo_estado, done)
        #
        # Depois:
        #
        # states
        # actions
        # rewards
        # next_states
        # dones
        states, actions, rewards, next_states, dones = zip(*batch)


        # ----------------------------------------------------
        # TRANSFORMAR OS DADOS EM TENSORS
        # ----------------------------------------------------

        # Estados atuais.
        b_s = torch.FloatTensor(
            np.array(states)
        )


        # Ações realizadas.
        #
        # LongTensor porque ações são índices inteiros.
        b_a = torch.LongTensor(actions).unsqueeze(1)


        # Recompensas recebidas.
        b_r = torch.FloatTensor(
            rewards
        ).unsqueeze(1)


        # Próximos estados.
        b_ns = torch.FloatTensor(
            np.array(next_states)
        )


        # Indica se o episódio terminou.
        #
        # False -> 0
        # True  -> 1
        b_d = torch.FloatTensor(
            dones
        ).unsqueeze(1)


        # ====================================================
        # 9. Q-VALUE QUE A REDE ATUAL PREVIU
        # ====================================================

        # A rede principal analisa os estados.
        #
        # Ela produz algo como:
        #
        # Estado 1 -> [0.7, 0.3]
        # Estado 2 -> [0.2, 0.8]
        #
        # Mas queremos somente o Q-value
        # da ação que realmente foi realizada.
        #
        # gather(1, b_a) pega exatamente esses valores.
        q_values = q_net(b_s).gather(1, b_a)


        # ====================================================
        # 10. CALCULAR O ALVO
        # ====================================================

        # Não queremos treinar a Rede Alvo.
        #
        # Ela serve apenas para produzir
        # uma referência estável.
        with torch.no_grad():

            # A Rede Alvo analisa os próximos estados.
            #
            # Exemplo:
            #
            # [0.5, 0.9]
            #
            # Queremos o maior:
            #
            # 0.9
            #
            # max(1, keepdim=True)[0]
            # pega o maior Q-value de cada estado.
            max_next_q = target_net(
                b_ns
            ).max(
                1,
                keepdim=True
            )[0]


            # =================================================
            # EQUAÇÃO DE BELLMAN
            # =================================================
            #
            # alvo =
            # recompensa
            # +
            # gamma * melhor valor futuro
            #
            # Porém:
            #
            # se o episódio acabou,
            # não existe futuro.
            #
            # Por isso usamos:
            #
            # (1 - b_d)
            #
            # Se b_d = 1:
            #
            # 1 - 1 = 0
            #
            # então:
            #
            # alvo = recompensa
            #
            # Se b_d = 0:
            #
            # 1 - 0 = 1
            #
            # então:
            #
            # alvo =
            # recompensa +
            # gamma * futuro
            targets = (
                b_r
                + (1 - b_d)
                * gamma
                * max_next_q
            )


        # ====================================================
        # 11. CALCULAR O ERRO
        # ====================================================

        # Comparamos:
        #
        # O que a rede disse
        #
        # contra
        #
        # O que ela deveria ter dito.
        #
        # Exemplo:
        #
        # q_values = 0.4
        # targets  = 0.8
        #
        # Existe um erro de 0.4.
        loss = nn.MSELoss()(
            q_values,
            targets
        )


        # ====================================================
        # 12. BACKPROPAGATION
        # ====================================================

        # Primeiro apagamos os gradientes anteriores.
        optimizer.zero_grad()


        # Calcula como cada peso da rede
        # contribuiu para o erro.
        loss.backward()


        # Agora o Adam modifica os pesos
        # tentando diminuir o erro.
        optimizer.step()


    # ========================================================
    # 13. ATUALIZAR A REDE ALVO
    # ========================================================

    # A cada 100 passos:
    #
    # copiamos os pesos da rede principal
    # para a rede alvo.
    if step % C == 0:

        target_net.load_state_dict(
            q_net.state_dict()
        )


# ============================================================
# 14. FINALIZAR
# ============================================================

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

---

## 💻 Código 5: Double DQN

O ponto principal do **Double DQN** é separar duas decisões que, no DQN comum, ficam misturadas:

1. a `q_net` **escolhe** a melhor ação;
2. a `target_net` **avalia** o valor dessa ação.

No DQN comum, poderíamos fazer diretamente:

```python
next_q = target_net(next_states).max(1, keepdim=True)[0]
```

No Double DQN, fazemos em duas etapas:

```python
next_actions = q_net(next_states).argmax(1, keepdim=True)
next_q = target_net(next_states).gather(1, next_actions)
```

Assim, a rede alvo não escolhe a ação que vai avaliar.

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# Exemplo: 4 informações no estado e 2 ações possíveis.
q_net = QNetwork(4, 2)
target_net = QNetwork(4, 2)

# Normalmente a rede alvo é uma cópia inicial da principal.
target_net.load_state_dict(q_net.state_dict())

states = torch.randn(8, 4)
next_states = torch.randn(8, 4)
rewards = torch.randn(8, 1)
dones = torch.zeros(8, 1)

gamma = 0.99

# Ações que realmente foram executadas.
actions = torch.randint(0, 2, (8, 1))
current_q = q_net(states).gather(1, actions)

with torch.no_grad():
    # 1. A REDE PRINCIPAL escolhe a melhor ação.
    next_actions = q_net(next_states).argmax(
        dim=1,
        keepdim=True
    )

    # 2. A REDE ALVO avalia a ação escolhida.
    next_q = target_net(next_states).gather(
        1,
        next_actions
    )

    # 3. Montamos o alvo de Bellman.
    target = rewards + (1 - dones) * gamma * next_q

loss = nn.MSELoss()(current_q, target)

print("Q atual:", current_q[:3])
print("Ação escolhida pela q_net:", next_actions[:3])
print("Target:", target[:3])
print("Loss:", loss.item())
```

### O que observar

Imagine que, para um próximo estado, a `q_net` estime:

```text
Ação 0 → 4.0
Ação 1 → 5.0
```

A `q_net` escolhe a ação `1`.

Agora suponha que a `target_net` estime:

```text
Ação 0 → 4.2
Ação 1 → 4.7
```

O Double DQN usa **4.7**, porque a `q_net` escolheu a ação `1` e a `target_net` apenas avaliou essa escolha.

No DQN convencional, a própria `target_net` faria o `max()` e escolheria a ação com base em suas próprias estimativas.

---

## 💻 Código 6: Dueling DQN

A arquitetura **Dueling** mantém uma parte da rede compartilhada e depois cria dois fluxos:

```text
Estado
  │
  ▼
Camadas compartilhadas
  ├──────────────► V(s)
  │
  └──────────────► A(s, a)
                       │
                       ▼
                     Q(s,a)
```

O código abaixo implementa a combinação:

```text
Q(s, a) = V(s) + A(s, a) - média(A(s, a))
```

```python
import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Parte compartilhada:
        # extrai características do estado.
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        # Value stream:
        # produz UM valor para o estado inteiro.
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream:
        # produz UM valor para cada ação.
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)

        # V(s)
        value = self.value_stream(features)

        # A(s,a)
        advantage = self.advantage_stream(features)

        # Q(s,a) = V(s) + A(s,a) - média das vantagens
        q_values = (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )

        return q_values


net = DuelingDQN(
    state_dim=4,
    action_dim=2
)

state = torch.randn(1, 4)

q_values = net(state)

print("Q-values:", q_values)
print("Melhor ação:", q_values.argmax(dim=1).item())
```

### O que acontece dentro da rede?

Se a rede produzir:

```text
V(s) = 10

A(s, 0) = -2
A(s, 1) = +2
```

A média das vantagens é `0`, então:

```text
Q(s, 0) = 10 - 2 = 8
Q(s, 1) = 10 + 2 = 12
```

A ideia é que a rede possa aprender separadamente:

- **"Este estado é bom ou ruim?"** → `V(s)`
- **"Qual ação é melhor ou pior neste estado?"** → `A(s,a)`

Isso é especialmente interessante quando várias ações são parecidas.

---

## 💻 Código 7: Distributional DQN (C51)

No **Distributional DQN**, a saída deixa de ser simplesmente:

```text
ação → Q-value
```

e passa a ser:

```text
ação → probabilidades sobre vários retornos
```

No C51, definimos previamente os _atoms_, que são os valores possíveis usados para representar a distribuição.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class C51Network(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51):
        super().__init__()

        self.action_dim = action_dim
        self.num_atoms = num_atoms

        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        # Cada ação terá uma distribuição com num_atoms probabilidades.
        self.output = nn.Linear(
            128,
            action_dim * num_atoms
        )

    def forward(self, x):
        features = self.feature(x)

        logits = self.output(features)

        # [batch, ações, atoms]
        logits = logits.view(
            -1,
            self.action_dim,
            self.num_atoms
        )

        # Softmax transforma os logits em probabilidades.
        probabilities = F.softmax(logits, dim=2)

        return probabilities


num_atoms = 51
v_min = -10
v_max = 10

# Valores que representam os possíveis retornos.
atoms = torch.linspace(
    v_min,
    v_max,
    num_atoms
)

net = C51Network(
    state_dim=4,
    action_dim=2,
    num_atoms=num_atoms
)

states = torch.randn(3, 4)

# [3 estados, 2 ações, 51 atoms]
distribution = net(states)

print("Formato:", distribution.shape)

# O valor esperado de cada ação é:
# Q(s,a) = soma(probabilidade * atom)
q_values = (distribution * atoms).sum(dim=2)

print("Q-values esperados:")
print(q_values)

# A ação escolhida continua podendo ser obtida
# pelo maior valor esperado.
actions = q_values.argmax(dim=1)

print("Melhores ações:", actions)
```

### Entendendo a saída

Suponha que uma ação possua:

```text
Atom 5  → 0.30
Atom 10 → 0.50
Atom 20 → 0.20
```

O valor esperado será:

```text
Q = 5(0.30) + 10(0.50) + 20(0.20)
Q = 1.5 + 5 + 4
Q = 10.5
```

Portanto, o Distributional DQN consegue representar a **distribuição** e ainda obter o Q-value tradicional através da média.

O exemplo acima mostra a cabeça de distribuição e o cálculo do valor esperado. Uma implementação completa do C51 também precisa realizar a **projeção da distribuição alvo** sobre os atoms fixos durante o treinamento.

---

## 💻 Código 8: Multi-step Learning

No DQN de um passo, o alvo é:

```text
rₜ + γ max Q(sₜ₊₁, a)
```

No exemplo abaixo vamos utilizar `n = 3`. Isso significa que acumulamos três recompensas antes de usar a estimativa do estado futuro.

```python
import torch

def calculate_n_step_target(rewards, next_q, gamma):
    """
    rewards:
        Lista com as recompensas dos n passos.

    next_q:
        Melhor Q-value estimado depois dos n passos.

    gamma:
        Fator de desconto.
    """

    # Começamos pela estimativa futura.
    target = next_q

    # Adicionamos as recompensas de trás para frente.
    for reward in reversed(rewards):
        target = reward + gamma * target

    return target


# Exemplo com 3 recompensas:
# r_t, r_t+1, r_t+2
rewards = [
    torch.tensor([[1.0]]),
    torch.tensor([[2.0]]),
    torch.tensor([[3.0]])
]

next_q = torch.tensor([[5.0]])

gamma = 0.99

target = calculate_n_step_target(
    rewards,
    next_q,
    gamma
)

print("Target de 3 passos:", target.item())
```

Uma forma mais direta de visualizar o mesmo cálculo é:

```python
r0 = 1.0
r1 = 2.0
r2 = 3.0
q3 = 5.0

target = (
    r0
    + gamma * r1
    + gamma**2 * r2
    + gamma**3 * q3
)

print(target)
```

### O que o código está fazendo?

Para `n = 3`, o alvo possui a estrutura:

```text
r0 + γr1 + γ²r2 + γ³Q(s3, a)
```

Ou seja, a atualização de `s0` já recebe informação de três recompensas futuras antes de depender apenas da estimativa da rede.

### Uma observação importante

Em uma implementação real, quando o episódio termina antes de completar `n` passos, não devemos inventar recompensas futuras. Nesse caso, o retorno é truncado no estado terminal.

---

## 💻 Código 9: Combinação das melhorias — Mini Rainbow

O **Rainbow DQN** combina diversas melhorias em um único agente. O objetivo deste exemplo é mostrar a arquitetura da combinação de forma didática.

Aqui vamos combinar diretamente:

- **Double DQN**;
- **Dueling Network**;
- **Distributional DQN (C51)**;
- **Multi-step Learning**.

O Rainbow completo também inclui **Prioritized Experience Replay** e **Noisy Networks**.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniRainbow(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51):
        super().__init__()

        self.action_dim = action_dim
        self.num_atoms = num_atoms

        # Parte compartilhada.
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Dueling + Distributional:
        # cada stream produz uma distribuição
        # sobre os atoms.
        self.value_stream = nn.Linear(128, num_atoms)

        self.advantage_stream = nn.Linear(
            128,
            action_dim * num_atoms
        )

    def forward(self, state):
        features = self.feature(state)

        # V(s), agora representado por atoms.
        value = self.value_stream(features)
        value = value.view(-1, 1, self.num_atoms)

        # A(s,a), também representado por atoms.
        advantage = self.advantage_stream(features)
        advantage = advantage.view(
            -1,
            self.action_dim,
            self.num_atoms
        )

        # Combinação Dueling:
        # Q = V + A - média(A)
        logits = (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )

        # Distribuição final.
        probabilities = F.softmax(logits, dim=2)

        return probabilities


# -----------------------------
# CONFIGURAÇÃO
# -----------------------------

state_dim = 4
action_dim = 2
num_atoms = 51

v_min = -10
v_max = 10

atoms = torch.linspace(v_min, v_max, num_atoms)

q_net = MiniRainbow(state_dim, action_dim, num_atoms)
target_net = MiniRainbow(state_dim, action_dim, num_atoms)

target_net.load_state_dict(q_net.state_dict())


# -----------------------------
# DOUBLE DQN + DISTRIBUTIONAL
# -----------------------------

states = torch.randn(8, state_dim)
next_states = torch.randn(8, state_dim)

with torch.no_grad():

    # 1. A q_net escolhe a ação usando
    # o valor esperado da distribuição.
    next_distribution = q_net(next_states)

    next_q = (next_distribution * atoms).sum(dim=2)

    next_actions = next_q.argmax(dim=1)

    # 2. A target_net avalia a ação escolhida.
    target_distribution = target_net(next_states)

    chosen_distribution = target_distribution[
        torch.arange(len(next_states)),
        next_actions
    ]

    print(
        "Distribuição escolhida:",
        chosen_distribution.shape
    )


# -----------------------------
# MULTI-STEP
# -----------------------------

gamma = 0.99
n = 3

# Três recompensas consecutivas.
r0 = torch.ones(8, 1)
r1 = torch.ones(8, 1)
r2 = torch.ones(8, 1)

# Parte conhecida do retorno n-step:
#
# r0 + γr1 + γ²r2
#
# Em uma implementação completa de Distributional DQN,
# a distribuição futura também é deslocada e projetada
# sobre os atoms.

multi_step_return = (
    r0
    + gamma * r1
    + gamma**2 * r2
)

print(
    "Retorno acumulado de 3 passos:",
    multi_step_return[:3]
)


# -----------------------------
# ESCOLHA DA AÇÃO
# -----------------------------

with torch.no_grad():

    distribution = q_net(states)

    # Esperança da distribuição.
    q_values = (distribution * atoms).sum(dim=2)

    actions = q_values.argmax(dim=1)

print("Q-values:", q_values[:3])
print("Ações escolhidas:", actions[:3])
```

### Como as melhorias se encaixam?

Podemos visualizar o fluxo:

```text
                    Estado
                      │
                      ▼
                Rede Dueling
                 ┌────┴────┐
                 ▼         ▼
               V(s)       A(s,a)
                 └────┬────┘
                      ▼
              Distribuição C51
                      │
                      ▼
                Q esperado
                      │
                ┌─────┴─────┐
                ▼           ▼
             Seleção      Avaliação
             q_net       target_net
                │           │
                └─────┬─────┘
                      ▼
                Alvo Double DQN
                      │
              + retorno n-step
                      │
                      ▼
                   Loss
                      │
                      ▼
                 Atualização
```

O ponto mais importante é perceber que **as melhorias não são cinco algoritmos completamente separados**. Elas modificam partes diferentes do mesmo processo:

```text
Double DQN
→ como escolhemos e avaliamos a próxima ação

Dueling
→ como a rede representa V(s) e A(s,a)

Distributional
→ o que a rede aprende sobre o retorno

Multi-step
→ quantas recompensas entram no alvo

Rainbow
→ combina várias dessas ideias
```

O exemplo é propositalmente didático. Uma implementação completa do Rainbow ainda precisaria integrar corretamente, no treinamento, componentes como **Prioritized Experience Replay**, **Noisy Networks**, a **loss/projeção do C51** e o tratamento completo do retorno multi-step.

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
- [14] Double DQN
- [15] Dueling Network Architecture
- [16] Distributional DQN
- [17] Multi-step Learning
- [18] Rainbow DQN e combinações de melhorias do DQN

[⬆️ Voltar ao índice](#-estudos-de-aprendizado-por-reforço-deep-rl)

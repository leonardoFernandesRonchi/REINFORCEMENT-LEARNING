# 📚 Dicionário de Símbolos Matemáticos para Deep RL

> **Guia de Referência Prática para Iniciação Científica e Pesquisa Acadêmica**

**Baseado no livro:** _An Introduction to Deep Reinforcement Learning_ — François-Lavet et al. (2018)

Este guia foi elaborado para estudantes de Iniciação Científica que estão iniciando seus estudos em **Aprendizado por Reforço Profundo (Deep RL)**.

A matemática dessa área combina conceitos de:

- Probabilidade
- Redes neurais
- Aprendizado de máquina
- Controle estocástico
- Processos de Decisão de Markov

A ideia deste material é funcionar como um **dicionário de consulta rápida**, traduzindo símbolos matemáticos para conceitos intuitivos de programação e ambientes de jogos.

---

# 📌 Índice

## 1. Machine Learning e Deep Learning

- [1.1. Vetor de Entrada — x](#11-vetor-de-entrada--x)
- [1.2. Vetor de Saída — y](#12-vetor-de-saída--y)
- [1.3. Matrizes de Pesos — W₁, W₂](#13-matrizes-de-pesos--w₁-w₂)
- [1.4. Vetores de Viés — b₁, b₂](#14-vetores-de-viés--b₁-b₂)
- [1.5. Camada Oculta — h](#15-camada-oculta--h)
- [1.6. Função de Ativação — A](#16-função-de-ativação--a)
- [1.7. Parâmetros do Modelo — θ](#17-parâmetros-do-modelo--θ)
- [1.8. Erro Empírico — Iₛ[f]](#18-erro-empírico--isf)
- [1.9. Erro Esperado — I[f]](#19-erro-esperado--if)
- [1.10. Erro de Generalização — G](#110-erro-de-generalização--g)
- [1.11. Taxa de Aprendizado — α](#111-taxa-de-aprendizado--α)
- [1.12. Gradiente — ∇θ](#112-gradiente--θ)

## 2. Aprendizado por Reforço e MDPs

- [2.1. Espaço de Estados — S](#21-espaço-de-estados--s)
- [2.2. Estados — s₀, sₜ, sₜ₊₁](#22-estados--s₀-sₜ-sₜ₊₁)
- [2.3. Espaço de Ações — A](#23-espaço-de-ações--a)
- [2.4. Ação — aₜ](#24-ação--aₜ)
- [2.5. Função de Transição — T](#25-função-de-transição--t)
- [2.6. Recompensa — rₜ](#26-recompensa--rₜ)
- [2.7. Fator de Desconto — γ](#27-fator-de-desconto--γ)
- [2.8. Observação — ωₜ](#28-observação--ωₜ)
- [2.9. Esperança Matemática — E](#29-esperança-matemática--e)
- [2.10. Política — π](#210-política--π)
- [2.11. Função de Valor — Vπ](#211-função-de-valor--vπ)
- [2.12. Função de Valor de Ação — Qπ](#212-função-de-valor-de-ação--qπ)
- [2.13. Função de Vantagem — Aπ](#213-função-de-vantagem--aπ)

## 3. Métodos Baseados em Valor e DQN

- [3.1. Operador de Bellman — B](#31-operador-de-bellman--b)
- [3.2. Q-Value Ótimo — Q\*](#32-q-value-ótimo--q)
- [3.3. Valor Alvo — YₖQ](#33-valor-alvo--yₖq)
- [3.4. Função de Perda DQN — L_DQN](#34-função-de-perda-dqn--ldqn)
- [3.5. Parâmetros da Target Network — θ⁻ₖ](#35-parâmetros-da-target-network--θₖ)
- [3.6. Exploração — ε](#36-exploração--ε)
- [3.7. Dataset / Replay Buffer — D](#37-dataset--replay-buffer--d)

## 4. Equações Fundamentais

- [4.1. Camada Oculta](#41-camada-oculta)
- [4.2. Camada de Saída](#42-camada-de-saída)
- [4.3. Gradiente Descendente](#43-gradiente-descendente)
- [4.4. Propriedade de Markov](#44-propriedade-de-markov)
- [4.5. Função de Valor V](#45-função-de-valor-v)
- [4.6. Função de Valor Q](#46-função-de-valor-q)
- [4.7. Equação de Bellman](#47-equação-de-bellman)
- [4.8. Política Ótima](#48-política-ótima)
- [4.9. Função de Vantagem](#49-função-de-vantagem)
- [4.10. Operador de Bellman](#410-operador-de-bellman)
- [4.11. Valor Alvo do Fitted Q-Learning](#411-valor-alvo-do-fitted-q-learning)
- [4.12. Função de Perda DQN](#412-função-de-perda-dqn)

---

# 1. Machine Learning e Deep Learning

Estes símbolos representam o fluxo de dados em uma rede neural tradicional e em camadas convolucionais.

No Deep RL, as redes neurais são utilizadas principalmente para estimar:

- Funções de valor
- Políticas
- Valores Q
- Probabilidades de ações

---

## 1.1. Vetor de Entrada — x

**Nome técnico:** Input Features

### O que significa?

Representa os valores ou características entregues ao modelo.

### Tradução prática

É o vetor que representa o estado ou as informações recebidas pelo agente.

Por exemplo, em um jogo:

```text
x = [posição_x, posição_y, velocidade]
```

Em uma imagem, `x` poderia representar os pixels da tela.

---

## 1.2. Vetor de Saída — y

**Nome técnico:** Output Values

### O que significa?

É o resultado do processamento da rede após as transformações realizadas pelas camadas.

### Tradução prática

Pode representar as estimativas finais produzidas pela rede.

Em um DQN, por exemplo:

```text
y = [Q(s, ação_1), Q(s, ação_2), Q(s, ação_3)]
```

Cada valor representa a estimativa de recompensa para uma ação.

---

## 1.3. Matrizes de Pesos — W₁, W₂

**Nome técnico:** Weight Matrices

### O que significa?

São os parâmetros multiplicativos de cada camada que conectam os neurônios.

### Tradução prática

Podemos imaginar os pesos como as **conexões da rede neural**.

Durante o treinamento, esses valores são modificados para melhorar as previsões.

---

## 1.4. Vetores de Viés — b₁, b₂

**Nome técnico:** Bias Vectors

### O que significa?

São termos adicionados à multiplicação das matrizes de pesos.

### Tradução prática

Permitem que a rede faça ajustes adicionais na transformação dos dados, não dependendo apenas dos valores de entrada.

---

## 1.5. Camada Oculta — h

**Nome técnico:** Hidden Layer

### O que significa?

É uma representação intermediária dos dados produzida durante o processamento da rede.

### Tradução prática

Podemos imaginar `h` como características que a IA detectou.

Por exemplo:

```text
"há um obstáculo à esquerda"
"há um inimigo próximo"
"o jogador está se aproximando da parede"
```

---

## 1.6. Função de Ativação — A

**Nome técnico:** Activation Function

### O que significa?

É uma função não linear aplicada ao resultado de uma transformação da rede.

Um exemplo comum é a **ReLU**.

```text
ReLU(x) = max(0, x)
```

### Por que é importante?

A função de ativação permite que a rede aprenda padrões complexos.

Sem não-linearidade, várias camadas lineares poderiam ser reduzidas a uma única transformação linear.

---

## 1.7. Parâmetros do Modelo — θ

**Nome técnico:** Parameters

### O que significa?

`θ` representa o conjunto de parâmetros aprendíveis do modelo.

Normalmente podemos pensar em:

```text
θ = {W, b}
```

### Tradução prática

São os "parafusos ajustáveis" da rede neural.

Durante o treinamento, o algoritmo modifica `θ` para melhorar o desempenho.

---

## 1.8. Erro Empírico — Iₛ[f]

**Nome técnico:** Empirical Error

### O que significa?

É o erro médio calculado utilizando um conjunto de dados de treinamento.

### Tradução prática

É a penalidade que o modelo tenta minimizar durante o treinamento.

---

## 1.9. Erro Esperado — I[f]

**Nome técnico:** Expected Error

### O que significa?

É o erro médio teórico do modelo considerando toda a distribuição real dos dados.

### Tradução prática

Representa o erro que a IA teria no mundo real.

Esse valor geralmente não pode ser calculado diretamente porque a distribuição real dos dados é desconhecida.

---

## 1.10. Erro de Generalização — G

**Nome técnico:** Generalization Error

### O que significa?

Representa a diferença entre o erro esperado e o erro empírico.

Uma forma simplificada de visualizar:

```text
G = Erro Esperado - Erro Empírico
```

### Tradução prática

Ajuda a representar o quanto o modelo está sofrendo de **overfitting**.

Se o erro de generalização for alto, o modelo pode ter aprendido muito bem os dados de treinamento, mas apresentar desempenho ruim em dados novos.

---

## 1.11. Taxa de Aprendizado — α

**Nome técnico:** Learning Rate

### O que significa?

`α` controla o tamanho do passo utilizado para atualizar os parâmetros durante o treinamento.

### Tradução prática

É o tamanho do passo que a rede dá durante a aprendizagem.

Uma atualização simplificada pode ser representada por:

```text
pesos = pesos - α × gradiente
```

Se `α` for muito grande:

```text
→ treinamento pode ficar instável
```

Se `α` for muito pequeno:

```text
→ treinamento pode ficar muito lento
```

---

## 1.12. Gradiente — ∇θ

**Nome técnico:** Gradient of the Parameters

### O que significa?

O gradiente representa o vetor de derivadas parciais do erro em relação aos parâmetros `θ`.

### Tradução prática

Podemos imaginar o gradiente como uma **bússola matemática**.

Ele indica a direção na qual o erro aumenta.

Para diminuir o erro, normalmente seguimos na direção oposta:

```text
-∇θ
```

---

[⬆️ Voltar ao índice](#-índice)

---

# 2. Aprendizado por Reforço e MDPs

Estes símbolos descrevem a formalização clássica de um **Processo de Decisão de Markov (MDP)**.

Um MDP permite representar matematicamente:

- O ambiente
- O agente
- Os estados
- As ações
- As recompensas
- As transições
- A tomada de decisões sequenciais

---

## 2.1. Espaço de Estados — S

**Nome técnico:** State Space

### O que significa?

`S` representa o conjunto completo de todos os estados possíveis do ambiente.

### Tradução prática

Imagine um robô em um labirinto.

```text
S = {todas as posições possíveis do robô}
```

---

## 2.2. Estados — s₀, sₜ, sₜ₊₁

**Nome técnico:** States

Representam estados em diferentes momentos:

```text
s₀     → estado inicial
sₜ     → estado atual
sₜ₊₁   → próximo estado
```

### Tradução prática

Em um jogo:

```text
sₜ     = posição atual do jogador
sₜ₊₁   = posição depois da ação
```

---

## 2.3. Espaço de Ações — A

**Nome técnico:** Action Space

### O que significa?

`A` representa o conjunto de todas as ações possíveis que o agente pode realizar.

### Tradução prática

Em um jogo:

```text
A = {acelerar, frear, virar_esquerda, virar_direita}
```

---

## 2.4. Ação — aₜ

**Nome técnico:** Action

### O que significa?

`aₜ` representa a ação específica escolhida pelo agente no instante `t`.

### Tradução prática

Se o agente possui:

```text
A = {esquerda, direita, cima, baixo}
```

e escolhe `direita`:

```text
aₜ = direita
```

---

## 2.5. Função de Transição — T

**Nome técnico:** Transition Function

A função:

```text
T(s, a, s')
```

representa a probabilidade de chegar ao próximo estado `s'` ao executar a ação `a` no estado `s`.

### Tradução prática

É a "física" ou regra do ambiente.

Por exemplo:

```text
Estado atual:
s = grama molhada

Ação:
a = correr

Próximo estado:
s' = escorregou
```

O ambiente pode definir:

```text
P(s' | s, a) = 0.10
```

ou seja, existe uma probabilidade de 10% de ocorrer aquela transição.

---

## 2.6. Recompensa — rₜ

**Nome técnico:** Reward

### O que significa?

A recompensa é o sinal numérico imediato devolvido pelo ambiente após uma transição.

### Tradução prática

É como a pontuação do jogo.

Por exemplo:

```text
Comer fruta       → +10
Bater no obstáculo → -10
Movimento normal  → 0
```

A recompensa orienta o agente sobre quais comportamentos são desejáveis.

---

## 2.7. Fator de Desconto — γ

**Nome técnico:** Discount Factor

`γ` controla quanto as recompensas futuras importam.

Normalmente:

```text
0 ≤ γ ≤ 1
```

### Intuição

Quando:

```text
γ ≈ 0
```

O agente dá muita importância às recompensas imediatas.

Quando:

```text
γ ≈ 1
```

O agente dá mais importância às recompensas futuras.

### Tradução prática

Podemos pensar em `γ` como a **paciência da IA**.

---

## 2.8. Observação — ωₜ

**Nome técnico:** Observation

### O que significa?

`ωₜ` representa as informações que o agente consegue observar no instante `t`.

### Tradução prática

É aquilo que a IA "vê".

Em um ambiente totalmente observável:

```text
ωₜ = sₜ
```

Ou seja, a observação contém todas as informações necessárias para representar o estado.

---

## 2.9. Esperança Matemática — E

**Nome técnico:** Expected Value

A esperança matemática é representada por:

```text
E[...]
```

### O que significa?

Representa uma média ponderada considerando diferentes possibilidades de resultados.

### Tradução prática

Imagine jogar o mesmo jogo milhares de vezes.

A esperança representa aproximadamente:

```text
"Qual pontuação média espero obter?"
```

---

## 2.10. Política — π

**Nome técnico:** Policy

A política pode ser representada como:

```text
π(s)
```

ou:

```text
π(s, a)
```

### O que significa?

A política define o comportamento do agente.

Ela determina quais ações devem ser tomadas em determinados estados.

### Tradução prática

É o **manual de comportamento do agente**.

Por exemplo:

```text
Se inimigo estiver perto → fugir
Se objetivo estiver perto → avançar
Se obstáculo estiver à esquerda → virar direita
```

---

## 2.11. Função de Valor — Vπ(s)

**Nome técnico:** V-Value Function

A função:

```text
Vπ(s)
```

representa o retorno acumulado esperado começando no estado `s` e seguindo a política `π`.

### Tradução prática

Podemos pensar:

> "Se eu estou nesse estado e continuar seguindo minha estratégia, quanto espero ganhar no futuro?"

---

## 2.12. Função de Valor de Ação — Qπ(s, a)

**Nome técnico:** Q-Value Function

A função:

```text
Qπ(s, a)
```

representa o retorno acumulado esperado quando:

1. Estamos no estado `s`;
2. Escolhemos a ação `a`;
3. Depois seguimos a política `π`.

### Tradução prática

A pergunta seria:

> "Se eu estou aqui e fizer essa ação agora, quanto espero ganhar no futuro?"

---

## 2.13. Função de Vantagem — Aπ(s, a)

**Nome técnico:** Advantage Function

A função de vantagem é:

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

### O que significa?

Mede o quão boa ou ruim é uma ação em relação ao comportamento médio esperado naquele estado.

Se:

```text
Aπ(s,a) > 0
```

A ação é melhor que o valor médio esperado.

Se:

```text
Aπ(s,a) < 0
```

A ação é pior que o esperado.

---

[⬆️ Voltar ao índice](#-índice)

---

# 3. Métodos Baseados em Valor e DQN

Estes símbolos e equações descrevem algoritmos que estimam valores `Q` para aprender comportamentos ótimos.

Entre eles:

- Q-Learning
- Fitted Q-Learning
- Deep Q-Networks (DQN)

---

## 3.1. Operador de Bellman — B

**Nome técnico:** Bellman Operator

O operador de Bellman é um mapeamento que atualiza e refina uma função de valor.

### Tradução prática

É uma forma abstrata de representar a regra de atualização do Q-Learning.

Podemos pensar nele como o **coração matemático do algoritmo**.

---

## 3.2. Q-Value Ótimo — Q\*(s, a)

O valor:

```text
Q*(s, a)
```

representa o valor ótimo de executar a ação `a` no estado `s`.

### Tradução prática

É o valor de utilidade que o agente tenta aprender para escolher ações da melhor maneira possível.

A política ótima pode ser obtida através de:

$$
\pi^*(s) = \arg\max_{a \in A} Q^*(s,a)
$$

---

## 3.3. Valor Alvo — YₖQ

**Nome técnico:** Target Value

O valor alvo representa a meta que o modelo tenta atingir durante uma iteração do Fitted Q-Learning.

Uma forma de representá-lo é:

$$
Y_k^Q = r + \gamma \max_{a' \in A} Q(s',a';\theta_k)
$$

### Tradução prática

É aproximadamente:

```text
recompensa imediata
+
recompensa futura estimada
```

---

## 3.4. Função de Perda DQN — L_DQN

**Nome técnico:** Square Loss

Representa o erro quadrático entre:

```text
Q estimado pela rede
```

e:

```text
valor alvo
```

A função de perda pode ser representada por:

$$
L_{DQN} =
\left(
Q(s,a;\theta_k) - Y_k^Q
\right)^2
$$

### Tradução prática

É a função de custo que o otimizador do PyTorch tenta minimizar para ajustar os pesos da rede.

---

## 3.5. Parâmetros da Target Network — θ⁻ₖ

**Nome técnico:** Target Network Parameters

Representa os parâmetros da rede alvo.

A Target Network é uma cópia da rede principal que é atualizada periodicamente.

### Tradução prática

Em vez de alterar a rede alvo a cada passo, podemos mantê-la congelada durante várias iterações.

Depois de `C` iterações:

```text
θ⁻ ← θ
```

Isso ajuda a estabilizar o treinamento.

---

## 3.6. Exploração — ε

**Nome técnico:** Epsilon

`ε` representa a probabilidade de escolher uma ação aleatória em uma política ε-greedy.

Por exemplo:

```text
ε = 0.1
```

significa:

```text
10% → explorar
90% → utilizar a melhor ação conhecida
```

### Exploração vs. Exploitação

```text
Exploração
    ↓
Experimentar ações novas

Exploração
    ↓
Utilizar aquilo que já parece funcionar
```

O equilíbrio entre explorar e explorar é fundamental no aprendizado por reforço.

---

## 3.7. Dataset / Replay Buffer — D

**Nome técnico:** Dataset / Experience Replay

`D` representa o conjunto de experiências armazenadas pelo agente.

Uma experiência pode ser representada como:

```text
(s, a, r, s')
```

onde:

```text
s  → estado
a  → ação
r  → recompensa
s' → próximo estado
```

### Tradução prática

No DQN, isso normalmente corresponde à **Replay Memory** ou **Replay Buffer**.

O agente armazena experiências e posteriormente seleciona amostras aleatórias para treinar a rede.

---

[⬆️ Voltar ao índice](#-índice)

---

# 4. Equações Fundamentais

Esta seção reúne as principais equações apresentadas no material.

---

## 4.1. Camada Oculta

**Fórmula 2.6**

$$
h = A(W_1 \cdot x + b_1)
$$

Onde:

```text
h  → saída da camada oculta
A  → função de ativação
W₁ → matriz de pesos
x  → vetor de entrada
b₁ → vetor de viés
```

### Fluxo

```text
x
 ↓
W₁ × x
 ↓
+ b₁
 ↓
A(...)
 ↓
h
```

---

## 4.2. Camada de Saída

**Fórmula 2.7**

$$
y = W_2 \cdot h + b_2
$$

Onde:

```text
y  → saída da rede
W₂ → pesos da camada de saída
h  → saída da camada oculta
b₂ → viés da camada de saída
```

---

## 4.3. Gradiente Descendente

**Fórmula 2.8**

$$
\theta \leftarrow
\theta - \alpha \nabla_{\theta} I_S[f]
$$

Onde:

```text
θ              → parâmetros do modelo
α              → taxa de aprendizado
∇θ             → gradiente
Iₛ[f]          → erro empírico
```

### Intuição

```text
Parâmetros antigos
        ↓
calcula gradiente
        ↓
move na direção oposta
        ↓
novos parâmetros
```

---

## 4.4. Propriedade de Markov

**Definição 3.1**

$$
P(\omega_{t+1} \mid \omega_t, a_t)
=
P(\omega_{t+1}
\mid
\omega_t, a_t, \ldots, \omega_0, a_0)
$$

### Intuição

O próximo estado depende do estado atual e da ação atual, não sendo necessário consultar todo o histórico.

Em termos simplificados:

```text
Presente
   ↓
decisão
   ↓
próximo estado
```

---

## 4.5. Função de Valor V

**Fórmula 3.1**

$$
V^\pi(s)
=
E
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\mid
s_t=s,\pi
\right]
$$

### Intuição

Representa quanto retorno esperamos obter a partir do estado `s` seguindo a política `π`.

---

## 4.6. Função de Valor Q

**Fórmula 3.3**

$$
Q^\pi(s,a)
=
E
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\mid
s_t=s,
a_t=a,
\pi
\right]
$$

### Diferença entre V e Q

```text
V(s)
↓
"Quão bom é estar neste estado?"

Q(s,a)
↓
"Quão bom é executar esta ação neste estado?"
```

---

## 4.7. Equação de Bellman

**Fórmula 3.4**

$$
Q^\pi(s,a)
=
\sum_{s' \in S}
T(s,a,s')
\left[
R(s,a,s')
+
\gamma
Q^\pi(s',a'=\pi(s'))
\right]
$$

### Intuição

O valor atual pode ser entendido como:

```text
recompensa imediata
+
valor futuro descontado
```

---

## 4.8. Política Ótima

**Fórmula 3.6**

$$
\pi^*(s)
=
\arg\max_{a \in A}
Q^*(s,a)
$$

### Intuição

A política ótima escolhe a ação que possui o maior valor Q.

```text
Q(s, esquerda) = 5
Q(s, direita)  = 8
Q(s, cima)     = 3

        ↓

π*(s) = direita
```

---

## 4.9. Função de Vantagem

**Fórmula 3.7**

$$
A^\pi(s,a)
=
Q^\pi(s,a)
-
V^\pi(s)
$$

### Intuição

Compara uma ação específica com o valor médio esperado daquele estado.

```text
A > 0 → ação melhor que o esperado
A = 0 → ação equivalente ao esperado
A < 0 → ação pior que o esperado
```

---

## 4.10. Operador de Bellman

**Fórmula 4.2**

$$
(BK)(s,a)
=
\sum_{s' \in S}
T(s,a,s')
\left[
R(s,a,s')
+
\gamma
\max_{a' \in A}
K(s',a')
\right]
$$

### Intuição

O operador de Bellman atualiza uma estimativa de valor considerando:

```text
transição
    +
recompensa
    +
melhor valor futuro
```

---

## 4.11. Valor Alvo do Fitted Q-Learning

**Fórmula 4.3**

$$
Y_k^Q
=
r
+
\gamma
\max_{a' \in A}
Q(s',a';\theta_k)
$$

### Intuição

O alvo é:

```text
Y = recompensa atual
    +
    γ × melhor Q do próximo estado
```

---

## 4.12. Função de Perda DQN

**Fórmula 4.4**

$$
L_{DQN}
=
\left(
Q(s,a;\theta_k)
-
Y_k^Q
\right)^2
$$

### Intuição

A rede tenta fazer:

```text
Q estimado ≈ valor alvo
```

Quanto maior a diferença, maior será a perda.

---

# 📌 Resumo dos Símbolos

| Símbolo   | Nome                 | Significado                 |
| --------- | -------------------- | --------------------------- |
| `x`       | Input                | Vetor de entrada            |
| `y`       | Output               | Vetor de saída              |
| `W₁, W₂`  | Weights              | Matrizes de pesos           |
| `b₁, b₂`  | Bias                 | Vetores de viés             |
| `h`       | Hidden Layer         | Representação intermediária |
| `A`       | Activation           | Função de ativação          |
| `θ`       | Parameters           | Parâmetros do modelo        |
| `Iₛ[f]`   | Empirical Error      | Erro empírico               |
| `I[f]`    | Expected Error       | Erro esperado               |
| `G`       | Generalization Error | Erro de generalização       |
| `α`       | Learning Rate        | Taxa de aprendizado         |
| `∇θ`      | Gradient             | Gradiente                   |
| `S`       | State Space          | Espaço de estados           |
| `sₜ`      | State                | Estado atual                |
| `A`       | Action Space         | Espaço de ações             |
| `aₜ`      | Action               | Ação atual                  |
| `T`       | Transition           | Função de transição         |
| `rₜ`      | Reward               | Recompensa                  |
| `γ`       | Discount Factor      | Fator de desconto           |
| `ωₜ`      | Observation          | Observação                  |
| `E`       | Expected Value       | Esperança matemática        |
| `π`       | Policy               | Política                    |
| `Vπ(s)`   | V-Value              | Valor do estado             |
| `Qπ(s,a)` | Q-Value              | Valor da ação               |
| `Aπ(s,a)` | Advantage            | Vantagem                    |
| `B`       | Bellman Operator     | Operador de Bellman         |
| `Q*(s,a)` | Optimal Q-Value      | Q-Value ótimo               |
| `YₖQ`     | Target Value         | Valor alvo                  |
| `L_DQN`   | DQN Loss             | Função de perda             |
| `θ⁻ₖ`     | Target Parameters    | Parâmetros da rede alvo     |
| `ε`       | Epsilon              | Exploração                  |
| `D`       | Dataset              | Replay Buffer               |

---

# 🧠 Mapa Mental

```text
APRENDIZADO POR REFORÇO
│
├── AMBIENTE
│   ├── S → Estados possíveis
│   ├── s → Estado atual
│   ├── A → Ações possíveis
│   ├── a → Ação escolhida
│   ├── T → Transição
│   └── r → Recompensa
│
├── POLÍTICA
│   └── π → Estratégia do agente
│
├── VALORES
│   ├── Vπ(s) → Valor do estado
│   ├── Qπ(s,a) → Valor da ação
│   └── Aπ(s,a) → Vantagem
│
├── DQN
│   ├── Q(s,a;θ) → Rede principal
│   ├── θ⁻ → Target Network
│   ├── Y → Valor alvo
│   ├── L → Função de perda
│   ├── D → Replay Buffer
│   └── ε → Exploração
│
└── APRENDIZADO
    ├── α → Learning Rate
    └── ∇θ → Gradiente
```

---

# 📚 Referência

François-Lavet et al. (2018).

_An Introduction to Deep Reinforcement Learning._

---

[⬆️ Voltar ao índice](#-índice)

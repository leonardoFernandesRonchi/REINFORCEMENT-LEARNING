# 📚 Capítulo 7: O Conceito de Generalização

## 📌 Visão Geral e Definição

No Aprendizado por Reforço (RL), a **generalização** refere-se a duas capacidades fundamentais do agente:

1. **Desempenho em dados limitados (_Sample Efficiency_)**: capacidade de obter bom desempenho em um ambiente de teste idêntico ao de treinamento, mas onde apenas um conjunto limitado de dados foi coletado. Isso é especialmente relevante quando o espaço de estados e ações é grande demais para ser completamente visitado.

2. **Desempenho em ambientes correlacionados**: capacidade de obter bom desempenho em um ambiente de teste que compartilha padrões com o ambiente de treino, mas apresenta variações nas dinâmicas ou nas recompensas. Esse conceito está associado a **transfer learning** e **meta-learning**.

---

## ⚖️ A Decomposição da Subotimalidade e o Trade-off Viés–Overfitting

Ao considerar um conjunto de dados finito $\mathcal{D}$ composto por quádruplas de transição

$$
\langle s, a, r, s' \rangle
$$

amostradas de forma independente e identicamente distribuída (i.i.d.), o algoritmo de aprendizado mapeia esse conjunto de dados em uma política $\pi_{\mathcal{D}}$.

A subotimalidade do retorno esperado dessa política pode ser decomposta em dois componentes principais:

$$
\mathbb{E}_{\mathcal{D} \sim D}
\left[
V^{\pi^*}(s) - V^{\pi_{\mathcal{D}}}(s)
\right]
=
\underbrace{
\left(
V^{\pi^*}(s) - V^{\pi_{\mathcal{D}_\infty}}(s)
\right)
}_{\text{Viés Assintótico}}
+
\underbrace{
\mathbb{E}_{\mathcal{D} \sim D}
\left[
V^{\pi_{\mathcal{D}_\infty}}(s) - V^{\pi_{\mathcal{D}}}(s)
\right]
}_{\text{Erro de Overfitting}}
$$

### Viés Assintótico (_Asymptotic Bias_)

É o erro que permanece independentemente da quantidade de dados.

Pode ser causado por:

- simplificações no modelo;
- limitações na classe de políticas;
- premissas incorretas do algoritmo.

### Erro de _Overfitting_ (Variância Paramétrica)

É o erro diretamente relacionado ao tamanho finito e limitado do conjunto de dados $\mathcal{D}$.

### Trade-off

O objetivo de um algoritmo de RL é minimizar a subotimalidade total encontrando um equilíbrio entre viés e _overfitting_.

#### Dados limitados / baixa qualidade

O algoritmo deve favorecer políticas mais robustas, utilizando uma classe menor de políticas com maior capacidade de generalização.

Isso significa aceitar determinado nível de **viés** para reduzir o risco de **overfitting**.

#### Dados abundantes / alta qualidade

Quando há muitos dados confiáveis, o risco de _overfitting_ diminui significativamente.

Assim, o algoritmo pode confiar mais nas estatísticas dos dados e utilizar modelos mais complexos, reduzindo o **viés assintótico**.

---

# 🧩 Exemplo Ilustrativo: Mundo em Grade 3×3 (_Grid World_)

O capítulo exemplifica o problema do _overfitting_ em dados limitados utilizando um MDP com:

- $N_S = 11$ estados;
- $N_A = 4$ ações;
- uma grade $3 \times 3$.

A partir do estado central:

- mover-se para a **esquerda** conduz deterministicamente a um estado que fornece recompensa de **0,6** a cada ciclo;
- mover-se para a **direita** conduz a um estado com **25% de probabilidade** de obter uma recompensa de **1,0**.

Com conhecimento perfeito do ambiente, a política ótima é ir repetidamente para a esquerda.

Porém, se o agente tiver apenas **uma amostra de experiência por par estado–ação**, as estatísticas frequentistas apresentam aproximadamente **58% de chance** de indicar erroneamente que a ação para a direita oferece acesso determinístico à recompensa de 1,0.

Assim, um algoritmo sem mecanismos de generalização pode escolher a decisão subótima.

---

# 🛠️ Os 4 Pilares para Controlar e Melhorar a Generalização

## 1. Seleção de Características (_Feature Selection_)

A escolha do nível adequado de abstração da representação do estado afeta diretamente o equilíbrio entre viés e _overfitting_.

### Risco de _Overfitting_

Incluir características em excesso, como a coordenada $y$ no exemplo da grade, pode fazer o algoritmo aprender correlações espúrias presentes nos dados amostrais.

### Risco de Viés Assintótico

Remover características essenciais que discriminam estados com papéis distintos na dinâmica impede que o agente diferencie esses estados.

Isso pode forçar a utilização de uma política subótima única.

### Representações Abstratas em Deep RL

Arquiteturas **encoder-decoder**, como **autoencoders**, podem ajudar a extrair fatores gerativos.

Porém, focar exclusivamente na reconstrução de pixels pode:

- preservar detalhes visuais irrelevantes para a tarefa;
- descartar detalhes muito pequenos, mas importantes para a tomada de decisão.

---

## 2. Escolha do Algoritmo e do Aproximador de Funções

A capacidade e a estrutura do aproximador de funções, como uma rede neural, determinam como as características são transformadas em abstrações de nível superior.

### Aproximadores muito simples

Podem aumentar o **viés assintótico**, pois possuem capacidade insuficiente para representar a solução necessária.

### Aproximadores muito complexos

Podem apresentar baixa capacidade de generalização e sofrer com **overfitting** quando os dados são limitados.

### Mecanismos de Atenção

Mecanismos de atenção podem atuar como uma forma de **seleção implícita de características**, permitindo que a rede dê mais importância às informações relevantes.

### Aprendizado Relacional e Raciocínio Simbólico

Estruturas de aprendizado relacional e raciocínio simbólico podem promover abstrações que facilitam a generalização.

### 2.1. Tarefas Auxiliares

Uma estratégia consiste em treinar a rede neural simultaneamente para realizar tarefas auxiliares.

Exemplos:

- prever recompensas imediatas com $\gamma = 0$;
- prever mudanças nos pixels;
- prever ativações ocultas.

Essas tarefas introduzem um **viés indutivo útil**, ajudando a reduzir o _overfitting_.

O agente **CRAR** é um exemplo dessa abordagem, construindo um espaço latente de baixa dimensão que combina componentes **model-free** e **model-based**.

---

# 3. Modificação da Função Objetivo

Modificar a função objetivo original permite introduzir um viés controlado para guiar e estabilizar o aprendizado.

## 3.1. Ajuste de Recompensa (_Reward Shaping_)

O _reward shaping_ adiciona uma função de recompensa intermediária:

$$
F(s, a, s')
$$

Essa função pode incorporar conhecimento prévio para acelerar o aprendizado em ambientes onde as recompensas são:

- esparsas;
- atrasadas;
- difíceis de obter diretamente.

---

## 3.2. Ajuste do Fator de Desconto ($\gamma$)

Reduzir artificialmente o fator de desconto $\gamma$ durante o treinamento altera o objetivo original, introduzindo certo viés.

Porém, isso pode:

- encurtar o horizonte de planejamento;
- reduzir o acúmulo de erros;
- diminuir o risco de _overfitting_ causado por erros nas estimativas de transição e recompensa.

Um $\gamma$ muito próximo de $1$ faz com que erros sejam acumulados ao longo das trajetórias simuladas.

Isso pode aumentar:

- a instabilidade;
- o viés de superestimação;
- os problemas em algoritmos de iteração de valor baseados em _bootstrapping_.

---

# 4. Aprendizado Hierárquico

O aprendizado com **opções (_options_)**, também chamadas de macro-ações ou ações abstratas, permite executar ações estendidas no tempo.

## Option-Critic

Arquiteturas como **Option-Critic** aprendem simultaneamente:

- as políticas internas de cada opção;
- as condições de término das opções;
- a política de alto nível responsável por escolher entre as opções.

Restringir o espaço de busca a comportamentos hierárquicos favorece políticas com propriedades reutilizáveis.

Isso pode melhorar a generalização em tarefas que envolvem escalas temporais longas.

---

# 🎯 Obtenção do Melhor Trade-off na Prática

## 5.1. Regime Batch (_Offline_)

No cenário **offline**, o equilíbrio entre viés e _overfitting_ pode ser avaliado utilizando um conjunto de validação de trajetórias separado do conjunto de treinamento.

### Ajuste de MDP Empírico

Constrói-se um modelo do MDP a partir dos dados utilizando métodos como:

- regressão;
- estatísticas frequentistas.

Esse modelo pode então ser utilizado para avaliar uma política.

### Estimadores _Model-Free Monte Carlo-like_ (MFMC)

Avaliam uma política gerando trajetórias artificiais a partir dos dados, sem necessariamente construir um modelo explícito do ambiente.

### Amostragem de Importância (_Importance Sampling_)

É um estimador não enviesado para dados _off-policy_.

Porém, sua variância pode crescer exponencialmente com o horizonte de planejamento.

### Estimador Duplamente Robusto (_Doubly Robust_)

Combina:

- uma abordagem baseada em modelo;
- amostragem de importância.

O objetivo é obter estimativas não enviesadas com menor variância.

### Séries Temporais Exógenas

Quando o ambiente depende de sinais externos, como:

- clima;
- mercado financeiro;

o sinal temporal pode ser dividido em conjuntos distintos de:

- treinamento;
- validação.

Isso evita avaliar o agente utilizando exatamente os mesmos períodos usados no treinamento.

---

# 5.2. Regime Online

No cenário **online**, os dados chegam sequencialmente e o trade-off entre viés e _overfitting_ evolui dinamicamente durante o treinamento.

### Aumento gradual da capacidade

Pode-se aumentar progressivamente a capacidade do aproximador de funções conforme mais dados são acumulados.

### Aumento progressivo de $\gamma$

O fator de desconto pode ser aumentado gradualmente ao longo do aprendizado.

Isso permite começar com um horizonte de planejamento menor e expandi-lo conforme o agente aprende.

### Adaptação dinâmica da arquitetura

A arquitetura da rede neural ou o espaço de características também pode ser adaptado durante o treinamento.

Exemplos:

- transformações **Net2Net**;
- técnicas de regularização;
- alteração dinâmica do espaço de características.

---

# 🧠 Resumo do Capítulo

A **generalização em Reinforcement Learning** está relacionada à capacidade do agente de utilizar o que aprendeu para tomar boas decisões mesmo quando possui dados limitados ou encontra variações relacionadas ao ambiente de treinamento.

O problema central pode ser entendido através do equilíbrio entre:

$$
\boxed{\text{Viés Assintótico} \quad \leftrightarrow \quad \text{Overfitting}}
$$

Os quatro principais mecanismos discutidos para controlar esse equilíbrio são:

1. **Seleção de características**  
   Escolher uma representação que mantenha informações relevantes sem incluir detalhes desnecessários.

2. **Escolha do algoritmo e do aproximador de funções**  
   Utilizar uma capacidade de representação adequada ao volume e à qualidade dos dados.

3. **Modificação da função objetivo**  
   Utilizar técnicas como _reward shaping_ e ajuste de $\gamma$ para introduzir vieses úteis.

4. **Aprendizado hierárquico**  
   Utilizar opções e comportamentos abstratos para reduzir a complexidade da tomada de decisão.

Em termos gerais:

> **Poucos dados → modelos mais restritos e maior viés podem ajudar a evitar overfitting.**

> **Muitos dados → modelos mais flexíveis podem aproveitar melhor as informações disponíveis e reduzir o viés.**

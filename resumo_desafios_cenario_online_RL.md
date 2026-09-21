# Resumo — Desafios no Cenário Online em RL

## 1. Cenário online vs. offline

- **Offline/batch:** o conjunto de transições `(s, a, r, s')` é fixo.
- **Online:** o agente pode coletar novas experiências durante o aprendizado.
- No cenário online, dois pontos importantes são:
  - **Exploração vs. explotação:** decidir entre buscar novas informações ou aproveitar o conhecimento atual.
  - **Experience replay:** armazenar experiências para reutilizá-las durante o treinamento.

## 2. Exploração vs. explotação

- **Exploração:** obter novas informações sobre o ambiente, suas transições e recompensas.
- **Explotação:** utilizar o conhecimento atual para buscar o maior retorno esperado.
- O agente precisa equilibrar as duas estratégias.

### Dois cenários

- **Sem fase de treinamento separada:** existe um trade-off explícito; o desempenho é relacionado ao **cumulative regret**.
- **Com fase de treinamento:** o agente explora durante o treinamento e depois utiliza uma política de teste; o conceito associado é o **simple regret**.

## 3. Métodos de exploração

### Exploração não direcionada

Não utiliza conhecimento específico sobre a exploração do ambiente.

- **ε-greedy:** escolhe uma ação aleatória com probabilidade `ε` e a ação da política considerada ótima com probabilidade `1 - ε`.
- **Softmax/Boltzmann:** a probabilidade de escolher uma ação depende do retorno esperado.

### Exploração direcionada

Utiliza informações das experiências passadas para decidir onde explorar.

Pode usar:
- bônus de exploração;
- ganho de informação;
- estimativas de incerteza;
- medidas de novidade;
- modelos do ambiente e planejamento;
- demonstrações de especialistas.

Em espaços de alta dimensionalidade, a exploração se torna mais difícil.

## 4. Exploração em Deep RL

Algumas técnicas mencionadas:

- **DQN + ε-greedy:** a instabilidade natural da Q-network pode favorecer a exploração.
- **Bootstrapped DQN:** utiliza funções de valor aleatorizadas.
- **Dropout Q-network:** utiliza incerteza produzida pelo dropout.
- **Ruído paramétrico:** adiciona ruído aos pesos da rede.
- **Exploration bonuses:** recompensa estados considerados novos ou desconhecidos.
- **Pseudo-count:** estima quantas vezes ações foram realizadas em estados semelhantes.
- **Curiosity/novidade:** incentiva o agente a visitar estados pouco conhecidos.

## 5. Experience Replay

A **replay memory** armazena experiências passadas para que possam ser reutilizadas posteriormente.

Principais vantagens:
- melhora a **eficiência de dados**;
- permite reutilizar experiências;
- torna a distribuição dos mini-batches mais estável;
- pode ajudar na **convergência e estabilidade**;
- é especialmente adequada para métodos **off-policy**, como DQN.

A memória normalmente mantém um número limitado de experiências devido à capacidade de memória disponível.

## 6. Prioritized Experience Replay

Em vez de reproduzir todas as experiências com a mesma frequência, algumas recebem maior prioridade.

No texto, a prioridade é relacionada à magnitude do **TD error**.

- Transições consideradas mais inesperadas podem ser reproduzidas mais frequentemente.
- Isso pode melhorar o uso das experiências.
- Porém, o processo pode introduzir **viés** no retorno esperado.

Esse viés pode ser corrigido parcial ou completamente usando **weighted importance sampling**, especialmente próximo da convergência.

## Ideia principal

O cenário online de RL envolve principalmente duas questões:

1. **Como coletar experiências úteis?** → exploração vs. explotação.
2. **Como aproveitar melhor as experiências coletadas?** → experience replay e prioritized replay.

Em resumo:

**Exploração → coleta informações novas**

**Explotação → aproveita o que já foi aprendido**

**Experience Replay → reutiliza experiências passadas**

**Prioritized Replay → reutiliza mais frequentemente experiências consideradas importantes**

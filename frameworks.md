# 📚 Capítulo 9: Benchmarking Deep RL

## 📌 Visão Geral

A comparação de algoritmos de aprendizado por reforço profundo (Deep RL) é uma tarefa desafiadora devido à natureza estocástica do processo de aprendizado, à variação na escolha de hiperparâmetros e à estocasticidade inerente tanto ao ambiente quanto ao modelo [1].

Para garantir comparações justas, consistência e reprodutibilidade de resultados experimentais, a comunidade desenvolveu um conjunto de ambientes simulados padronizados (_benchmarks_) e diretrizes de avaliação [1].

---

## 🛠️ 9.1 Ambientes de Benchmark (_Benchmark Environments_)

Os ambientes de teste são divididos em três grandes categorias principais:

### 9.1.1 Problemas Clássicos de Controle (_Classic Control Problems_)

- **Tarefas Tradicionais:** Incluem o _Cartpole_ (equilibrar um bastão sobre um carrinho), o _Mountain Car_ (acelerar um carro para subir uma montanha usando momento) e o _Acrobot_ [2].
- **Uso:** Foram amplamente utilizados para algoritmos tabulares de RL e aproximação linear de funções, mas ainda são empregados para testar rapidamente algoritmos de Deep RL [2].

### 9.1.2 Jogos (_Games_)

Jogos de tabuleiro e videogames oferecem plataformas ricas para testar agentes em cenários com grandes espaços de estados/ações, dinâmicas frequentemente não-Markovianas e horizontes de planejamento de longo prazo com recompensas esparsas [3, 4]:

- **Jogos de Tabuleiro:** Xadrez, Go e Poker, onde o Deep RL alcançou nível super-humano [3].

- **Arcade Learning Environment (ALE):** Suíte baseada em jogos do Atari 2600, como _Pong_, _Asteroids_, _Space Invaders_, _Seaquest_, _Breakout_ e _Montezuma’s Revenge_ [4, 5]. Serve como padrão para avaliar o aprendizado a partir de pixels, além de testar generalização, aprendizado multitarefa e aprendizado por transferência [4, 5].

- **GVGAI (_General Video Game AI_):** Estrutura de competição desenvolvida para testar e comparar agentes em múltiplos jogos desconhecidos e na criação de regras e níveis [5, 6].

- **ViZDoom:** Simulador baseado no jogo _Doom_, utilizado para pesquisas sobre modificação de recompensa (_reward shaping_), aprendizado por currículo (_curriculum learning_), planejamento preditivo e _meta-RL_ [6].

- **Project Malmo (Minecraft):** Ambiente em mundo aberto para exploração de problemas de navegação, resolução colaborativa de problemas, aprendizado contínuo (_lifelong learning_) e planejamento hierárquico [6, 7].

- **DeepMind Lab:** Plataforma 3D em labirintos adaptada do jogo _Quake_, usada para aprendizado hierárquico e contínuo [7].

- **StarCraft II e StarCraft: Broodwar:** Jogos de estratégia em tempo real (RTS) que servem como ambientes para estudar sistemas multiagente, aprendizado hierárquico e planejamento complexo [7].

### 9.1.3 Sistemas de Controle Contínuo e Domínios de Robótica

Em muitas aplicações reais e na robótica, as ações tomadas pelo agente são contínuas, como a aplicação de torques em motores de juntas [8]:

- **MuJoCo:** Simulador físico amplamente utilizado para tarefas de locomoção e controle contínuo de robôs em espaço 3D [8]. Por ser uma ferramenta proprietária e exigir licença, impulsionou o desenvolvimento de alternativas abertas [9].

- **Roboschool:** Iniciativa de código aberto baseada em simulações semelhantes às do MuJoCo, incluindo robôs humanoides enfrentando obstáculos [9].

---

## 🎯 9.2 Melhores Práticas para Benchmarking em Deep RL

- **Reprodutibilidade e Variância:** Devido à alta sensibilidade das redes profundas a sementes aleatórias (_random seeds_), inicializações de pesos e variação de hiperparâmetros, o capítulo enfatiza a importância de reportar métricas agregadas sobre múltiplas execuções e manter o código e os ambientes rigorosamente padronizados [1].

---

## 💻 9.3 Softwares de Código Aberto para Deep RL

- O progresso na área é sustentado por ecossistemas e bibliotecas de código aberto, como o OpenAI Gym e implementações de referência, que facilitam a integração entre ambientes de simulação e algoritmos de aprendizado por reforço profundo [1].

## 💻 A.1 Frameworks de Deep RL

A seguir está uma lista de alguns frameworks conhecidos utilizados para Deep RL:

- **DeeR** (François-Lavet et al., 2016): tem como foco ser (i) facilmente acessível e (ii) modular para pesquisadores.

- **Dopamine** (Bellemare et al., 2018): fornece algoritmos padrão juntamente com _baselines_ para os jogos de Atari.

- **ELF** (Tian et al., 2017): é uma plataforma de pesquisa para Deep RL, voltada principalmente para jogos de estratégia em tempo real (_Real-Time Strategy — RTS_).

- **OpenAI Baselines** (Dhariwal et al., 2017): é um conjunto de algoritmos populares de Deep RL, incluindo **DDPG, TRPO, PPO e ACKTR**. O foco desse framework é fornecer implementações de _baselines_.

- **PyBrain** (Schaul et al., 2010): é uma biblioteca de aprendizado de máquina que possui suporte a aprendizado por reforço.

- **rllab** (Duan et al., 2016a): fornece um conjunto de implementações de algoritmos de Deep RL avaliadas por meio de benchmarks.

### 📊 Características dos Frameworks

| Framework            | Deep RL | Interface Python | Suporte automático a GPU |
| -------------------- | ------- | ---------------- | ------------------------ |
| **DeeR**             | Sim     | Sim              | Sim                      |
| **Dopamine**         | Sim     | Sim              | Sim                      |
| **ELF**              | Sim     | Não              | Sim                      |
| **OpenAI Baselines** | Sim     | Sim              | Sim                      |
| **PyBrain**          | Sim     | Sim              | Não                      |
| **RL-Glue**          | Não     | Sim              | Não                      |
| **RLPy**             | Não     | Sim              | Não                      |
| **rllab**            | Sim     | Sim              | Sim                      |
| **TensorForce**      | Sim     | Sim              | Sim                      |

**Tabela A.1:** Resumo de algumas características de frameworks de aprendizado por reforço existentes.

- **TensorForce** (Schaarschmidt et al., 2017): é um framework para Deep RL construído em torno do **TensorFlow**, contendo diversas implementações de algoritmos.

  Seu objetivo é mover os cálculos relacionados ao aprendizado por reforço para o grafo do TensorFlow, buscando ganhos de desempenho e eficiência. Por isso, ele possui forte dependência da biblioteca de Deep Learning TensorFlow.

  O framework fornece diversas implementações de algoritmos, incluindo **TRPO, DQN, PPO e A3C**.

### Frameworks gerais de Reinforcement Learning

Embora não tenham sido desenvolvidos especificamente para Deep RL, os dois frameworks a seguir também podem ser citados no contexto de aprendizado por reforço:

- **RL-Glue** (Tanner e White, 2009): fornece uma interface padrão que permite conectar agentes de RL, ambientes e programas de experimentos.

- **RLPy** (Geramifard et al., 2015): é um framework focado em RL baseado em valores (_value-based RL_), utilizando aproximadores de função lineares e ações discretas.

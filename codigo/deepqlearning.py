import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque



# Eu passo a quantia de estados e de ações, e ele retorna os Q values. É uma função pronta para redes neurais python
class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.fc = nn.Sequential(

            # Recebe 4 valores e transforma em 64 (já que temos 4 estados)
            nn.Linear(state_dim, 64),

            nn.ReLU(),

            # Transforma 64 valores em 2 (campo de ações, 2 ações nesse caso)

            nn.Linear(64, action_dim)
        )

    def forward(self, x):

        return self.fc(x)


# Cria o ambiente do jogo
env = gym.make("CartPole-v1")


#Pega os estados
state_dim = env.observation_space.shape[0]


#Pega as ações
action_dim = env.action_space.n


# Cria a rede
q_net = QNetwork(state_dim, action_dim)

#Cria a rede alvo
target_net = QNetwork(state_dim, action_dim)


#Pegue todos os pesos da q_net e copie para a target_net."
target_net.load_state_dict(q_net.state_dict())



optimizer = optim.Adam(
    q_net.parameters(),
    lr=0.001
)



replay_memory = deque(maxlen=10000)

# Valorização do futuro
gamma = 0.99

# Quantidade de experiencias utilizadas em cada treinamento
batch_size = 64


# A cada quantos passos atualizamos a Rede Alvo.
C = 100

# 10% de ações aleatorias
epsilon = 0.1


state, _ = env.reset()



for step in range(1, 1001):

#   Antes:

#  [0.1, 0.2, 0.3, 0.4]

#  Depois:

#  [[0.1, 0.2, 0.3, 0.4]]

    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    if random.random() < epsilon:

        action = env.action_space.sample()

    else:
       # q_net(state_tensor):
       # Passa o estado atual pela rede e retorna um Q-value para cada ação.
       # Exemplo: [0.72, 0.35] → ação 0 tem Q = 0.72 e ação 1 tem Q = 0.35.
       
       # torch.argmax():
       # Encontra o maior Q-value e retorna o índice dele.
       # Exemplo: argmax([0.72, 0.35]) → 0, então a ação escolhida é a 0.
       
       # .item():
       # Converte o Tensor do PyTorch em um número Python normal.
       # Exemplo: tensor(0) → 0.
       
       # torch.no_grad():
       # Impede o cálculo de gradientes, pois aqui estamos apenas escolhendo
       # uma ação e não treinando a rede.
        with torch.no_grad():
            action = torch.argmax(
                q_net(state_tensor)
            ).item()


    # retorna o proximo estado, recompensa, se terminou (falta de tempo)
    next_state, reward, terminated, truncated, _ = env.step(action)


    done = terminated or truncated

     #adicionando informacao na memoria
    replay_memory.append((
        state,
        action,
        reward,
        next_state,
        done
    ))


    # se terminou, reinicia o ambiente, se não, vai para a próxima ação
    if done:
        state, _ = env.reset()

    else:
        state = next_state


   # se for maior ou igual ao tamanho do batch, faz o treinamento da rede (é 64 nesse caso)
    if len(replay_memory) >= batch_size:

       # Pega 64 experiências aleatórias da memória de replay para treinar a rede.
        batch = random.sample(
            replay_memory,
            batch_size
        )
        
        # separa os estados, ações, recompensas, próximos estados e se terminou em listas separadas.

        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Converte essas listas em tensores do PyTorch para poder usar na rede neural

        # Converte a lista de estados em um tensor de float
        b_s = torch.FloatTensor(
            np.array(states)
        )
        
        # Converte a lista de ações em um tensor de long (inteiros) e adiciona uma dimensão extra para compatibilidade
        b_a = torch.LongTensor(actions).unsqueeze(1)

        # Converte a lista de recompensas em um tensor de float e adiciona uma dimensão extra para compatibilidade
        b_r = torch.FloatTensor(
            rewards
        ).unsqueeze(1)
        
        # Converte a lista de próximos estados em um tensor de float
        b_ns = torch.FloatTensor(
            np.array(next_states)
        )

        # Converte a lista de "dones" (se o episódio terminou) em um tensor de float e adiciona uma dimensão extra para compatibilidade
        b_d = torch.FloatTensor(
            dones
        ).unsqueeze(1)

        # É confuso mesmo, mas no python esse objeto age como uma função, então você pode chamá-lo como se fosse uma função. Aqui, estamos passando o batch de estados para a rede neural para obter os Q-values correspondentes.
        # Envia os estados para a q_net, obtém os Q-values de todas as ações e, com gather, seleciona o Q-value da ação que foi executada em cada experiência
        q_values = q_net(b_s).gather(1, b_a)


        with torch.no_grad():
            #aqui usamos a target_net que criamos anteriormente, a rede alvo
            # E passamos para ela os proximos estados (b_ns)
            # Para obter o Q-value máximo para cada próximo estado, usamos .max(1, keepdim=True)[0]
            max_next_q = target_net(
                b_ns
            ).max(
                1,
                keepdim=True
            )[0]
            
           # Calcula os alvos para o treinamento da rede neural
           # b_r recompensa recebida
           # b_d indica se o episódio terminou (1 se terminou, 0 caso contrário)
           # gamma é o fator de desconto para recompensas futuras
           # max_next_q é o Q-value máximo do próximo estado, obtido da rede alvo
            targets = (
                b_r
                + (1 - b_d)
                * gamma
                * max_next_q
            )
        #Faz o cálculo da perda (loss) entre os Q-values atuais e os alvos calculados.
        # A função de perda utilizada é a Mean Squared Error (MSE),
        # que mede a diferença quadrática média entre os valores previstos (q_values) e os valores reais (targets).
        # Em seguida, o otimizador é zerado, o gradiente é calculado com loss.backward()
        # , e os pesos da rede são atualizados com optimizer.step().
        loss = nn.MSELoss()(
            q_values,
            targets
        )
        #Limpa os gradientes anteriores
        optimizer.zero_grad()
        #Calcula os gradientes da perda em relação aos parâmetros da rede
        loss.backward()
        #Eleva os gradientes calculados para atualizar os pesos da rede neural, ajustando-os para minimizar a perda.
        optimizer.step()

     #Atualiza a rede alvo
    if step % C == 0:

        target_net.load_state_dict(
            q_net.state_dict()
        )



env.close()

print("DQN treinado com sucesso!")
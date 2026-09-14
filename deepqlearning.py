import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque



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
    
env = gym.make("CartPole-v1")
#
# 1. posição do carrinho
# 2. velocidade do carrinho
# 3. ângulo do bastão
# 4. velocidade angular do bastão
state_dim = env.observation_space.shape[0]



# 0 = esquerda
# 1 = direita
action_dim = env.action_space.n


q_net = QNetwork(state_dim, action_dim)


    

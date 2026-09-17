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

### O que acontece dentro da rede?

#Se a rede produzir:

#V(s) = 10

#A(s, 0) = -2
#A(s, 1) = +2

#A média das vantagens é `0`, então:

#Q(s, 0) = 10 - 2 = 8
#Q(s, 1) = 10 + 2 = 12

#A ideia é que a rede possa aprender separadamente:

#- **"Este estado é bom ou ruim?"** → `V(s)`
#- **"Qual ação é melhor ou pior neste estado?"** → `A(s,a)`

#Isso é especialmente interessante quando várias ações são parecidas.


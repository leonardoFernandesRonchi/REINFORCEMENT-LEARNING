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
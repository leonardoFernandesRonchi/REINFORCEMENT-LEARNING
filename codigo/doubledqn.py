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

# [
# [s1, s2, s3, s4],  # estado 1
# [s1, s2, s3, s4],  # estado 2
# [s1, s2, s3, s4],  # estado 3
# [s1, s2, s3, s4],  # estado 4
# [s1, s2, s3, s4],  # estado 5
# [s1, s2, s3, s4],  # estado 6
# [s1, s2, s3, s4],  # estado 7
# [s1, s2, s3, s4],  # estado 8
#]
states = torch.randn(8, 4)
next_states = torch.randn(8, 4)
rewards = torch.randn(8, 1)
dones = torch.zeros(8, 1)

gamma = 0.99

# radiint cria um tensor com numeros inteiros (que começa em 0, e é incluido, e 2, que não é incluido)
#tensor([
# [1],
# [0],
# [1],
# [1],
# [0],
# [0],
# [1],
# [0]
#])
actions = torch.randint(0, 2, (8, 1))
current_q = q_net(states).gather(1, actions)

with torch.no_grad():
    # 1. A REDE PRINCIPAL escolhe a melhor ação.
    #Enviando para a rede os proximos estados, a rede retorna os Q-values de cada ação.
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
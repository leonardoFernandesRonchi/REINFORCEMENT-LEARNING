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

Uma forma mais direta de visualizar o mesmo cálculo é:

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
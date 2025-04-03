import torch
from guided_optimization.heisenberg import heisenberg_hamiltonian

H = heisenberg_hamiltonian((2, 2), 1)

eigen_value, eigen_vector = torch.linalg.eigh(H)
print(eigen_value)

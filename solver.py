import time

import torch
import torch.nn

from guided_optimization.heisenberg import heisenberg_hamiltonian


class BooleanMLP(torch.nn.Module):
    def __init__(self, dim_input: int, dim_output: int, hidden_size: tuple[int, ...]):
        super().__init__()
        self.dim_input: int = dim_input
        self.dim_output: int = dim_output
        self.hidden_size: tuple[int, ...] = hidden_size
        self.depth: int = len(hidden_size)

        dimensions: list[int] = [dim_input] + list(hidden_size) + [dim_output]
        linears: list[torch.nn.Module] = [torch.nn.Linear(i, j) for i, j in zip(dimensions[:-1], dimensions[1:])]
        layers: list[torch.nn.Module] = [layer for linear in linears for layer in (linear, torch.nn.ReLU())][:-1]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        # Convert boolean to float, track operations if training
        return self.network(x.float())


def num2bin(num_tensor, length):
    bin_tensor = torch.zeros([*num_tensor.size(), length], dtype=torch.bool)
    for i in range(length):
        bin_tensor[..., i] = (num_tensor & 1).to(dtype=bool)
        num_tensor = num_tensor >> 1
    return bin_tensor


def bin2num(bin_tensor):
    *size, length = bin_tensor.size()
    num_tensor = torch.zeros(size, dtype=torch.int64)
    for i in range(length):
        num_tensor += bin_tensor[..., i].to(dtype=torch.int64) << i
    return num_tensor


sys_size = (3, 4)
J = 1
H = torch.real(heisenberg_hamiltonian(sys_size, J))
dim_input = sys_size[0] * sys_size[1]
dim_vect = 2 ** dim_input

model = BooleanMLP(dim_input, 2, (64, 32))  # 用var的时候相当于es不训练，就不改这里了 :D
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

cycle_count = 0
output_interval = 1
target_energy = -6.691680193512
threshold = 1e-8
t=time.time()
print(f"starting at {t}")

while True:
    # loss computation start
    Es_sum = torch.tensor(0.)
    Es2_sum = torch.tensor(0.)
    es_loss_sum = torch.tensor(0.)
    ws2_sum = torch.tensor(0.)
    for s in range(dim_vect):
        ws, es = model(num2bin(torch.tensor(s), dim_input))
        if ws == 0:
            continue

        nonzeros = H[:, s].abs() >= 1e-8
        loc = nonzeros.nonzero().squeeze(1)  # t.nonzero outputs tensors with size[n, *t.size()]
        h_vec = H[nonzeros, s].unsqueeze(1)

        w_vec = model(num2bin(loc, dim_input))[..., 0].unsqueeze(1)
        Es = (h_vec.T @ w_vec).squeeze() / ws

        Es_sum = Es_sum + ws ** 2 * Es
        Es2_sum = Es2_sum + ws ** 2 * Es ** 2
        es_loss_sum = es_loss_sum + ws ** 2 * (Es - es) ** 2
        ws2_sum = ws2_sum + ws ** 2

    EEs = Es_sum / ws2_sum
    EEs2 = Es2_sum / ws2_sum
    VEs = EEs2 - EEs ** 2
    # Ees_loss = torch.sqrt(es_loss_sum / ws2_sum)  # ??????????????
    Ees_loss = es_loss_sum / ws2_sum
    loss = EEs + 0 * Ees_loss + 0 * VEs
    # loss end

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cycle_count += 1

    if cycle_count % output_interval == 0:
        print(f"{cycle_count} cycles done. Current loss: {loss.item(): 4f}")  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if EEs - target_energy < threshold:
        print(f"Training done after {cycle_count} cycles. \nEnergy reached: {EEs}. \nLoss reached: {loss.item()}")
        break

print(f"Ended at {time.time()}")
print(f"Finished in {time.time()-t} secs")

with torch.no_grad():
    predicted = model(num2bin(torch.tensor(range(dim_vect)), dim_input))
    w, e = predicted[..., 0], predicted[..., 1]
    error = H @ w - target_energy * w

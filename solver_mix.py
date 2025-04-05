import torch
import torch.nn as nn
from guided_optimization.heisenberg import heisenberg_hamiltonian

import math


class BooleanMLP(nn.Module):
    def __init__(self, input_size=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),  # Parameters: weight [64, 10], bias [64]
            nn.ReLU(),
            nn.Linear(64, 32),  # Parameters: weight [32, 64], bias [32]
            nn.ReLU(),
            nn.Linear(32, 2)  # Parameters: weight [2, 32], bias [2]
        )

    def forward(self, x):
        # Convert boolean to float, track operations if training
        return self.network(x.float())


def num2bin(num_tensor, length):
    bin_tensor = torch.zeros([*num_tensor.size(), length], dtype=torch.int64)
    for i in range(length):
        bin_tensor[...,i]^=num_tensor&1
        num_tensor >>= 1
    return bin_tensor>0.5

def bin2num(bin_tensor):
    bin_tensor=bin_tensor.int()
    *size, length=bin_tensor.size()
    num_tensor=torch.zeros(size, dtype=torch.int64)
    for i in range(length):
        num_tensor+=bin_tensor[...,i]<<i
    return num_tensor


H = torch.real(heisenberg_hamiltonian((2, 2), 1))
vec_dim = H.size()[0]
input_size = round(math.log(vec_dim, 2))
alpha=1

model = BooleanMLP(input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


cycle_count = 0
output_interval=10
loss_sum=0
for _ in range(1000):
    cycle_count+=1
    s_sample = range(vec_dim)  # definite version

    # loss computation start
    Es_sum=torch.tensor(0.)
    es_loss_sum=torch.tensor(0.)
    ws2_sum=torch.tensor(0.)
    for s in s_sample:
        # filter non-zero values from H[...,s] start
        loc, hvec = [], []
        for i in range(vec_dim):
            if abs(H[i,s])>=1e-8:
                loc.append(i)
                hvec.append(H[i, s])
        loc=torch.tensor(loc, dtype=torch.int64)
        hvec=torch.tensor(hvec).unsqueeze(1)
        # filter end

        wvec=model(num2bin(loc, input_size))[...,0].unsqueeze(1)
        ws, es=model(num2bin(torch.tensor(s), input_size))
        Es=(hvec.T@wvec).squeeze()/ws
        Es_sum+=ws**2*Es
        es_loss_sum+=ws**2*(Es-es)**2
        ws2_sum+=ws**2
    EEs=Es_sum/ws2_sum
    Ees_loss=es_loss_sum/ws2_sum
    loss=EEs+alpha*Ees_loss
    loss_sum+=loss.item()
    # loss end

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if cycle_count%output_interval==0:
        loss_avg=loss_sum / output_interval
        loss_sum=0
        print(f"{cycle_count} cycles done. Avg loss: {loss_avg: 4f}")
    if loss.item()+2<1e-7:
        break

predicted=model(num2bin(torch.tensor(range(vec_dim)),input_size))
w, e=predicted[...,0], predicted[...,1]
print(H@w/2+w)

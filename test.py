import torch
from guided_optimization.heisenberg import heisenberg_hamiltonian

def E_function(H, V):
    """
    H是2^n x 2^n的矩阵,
    V是2^n x 1的列向量
    """
    return torch.matmul(V.T,torch.matmul(H, V))/torch.matmul(V.T, V)

def grad(lr, V):
    """
    求E的梯度
    """
    with torch.no_grad():  
        V -= lr * V.grad #如果使用V = V - lr * V.grad,则会更新V的id
        V.grad.zero_()

m, n = 2, 2
H = torch.real(heisenberg_hamiltonian((m, n), 1))
print(torch.min(torch.real(torch.linalg.eigvals(H))))
V = torch.rand(2**(n*m), 1, requires_grad=True)#定义一个随机的2**(m*n)x1列向量

#超参数设置
lr = 10
num_epochs = 50
loss_function = E_function
way = grad

#训练过程
for epoch in range(num_epochs):
    loss = loss_function(H, V)
    loss.backward()
    way(lr, V)
    with torch.no_grad():
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
import time
import torch
import torch.nn as nn
from guided_optimization.heisenberg import heisenberg_hamiltonian

# 定义一个继承自nn.Module的VectorModel类
class VectorModel(nn.Module):
    def __init__(self, dim_input, hidden_sizes):
        super().__init__()
        layers = []
        prev_size = dim_input
        # 根据hidden_sizes动态添加线性层和ReLU激活函数
        for size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, size), nn.ReLU()])
            prev_size = size
        # 添加最终的线性层，输出每个基态的振幅
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.float()  # 确保输入数据为浮点数
        return self.net(x).squeeze(-1)  # 返回归一化的输出，形状 [batch_size]

# 将数字张量转换为指定长度的二进制编码函数
def num2bin(num_tensor, length):
    return (num_tensor.unsqueeze(-1).bitwise_and(1 << torch.arange(length)) > 0).squeeze()

sys_size = (3, 4)
J = 1
H = torch.real(heisenberg_hamiltonian(sys_size, J))
dim_input = sys_size[0] * sys_size[1]
dim_vect = 2 ** dim_input

# 生成所有可能状态的二进制编码
all_states = num2bin(torch.arange(dim_vect), dim_input)

# 初始化模型，输入维度为dim_input，隐藏层大小为64和32
model = VectorModel(dim_input, (64, 32))
# 使用Adam优化器对模型参数进行优化，学习率为0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 设置训练参数
max_cycles = 10000  # 最大训练周期数
print_interval = 100  # 每100个周期打印一次信息
target_energy = -6.691680193512  # 目标能量值
threshold = 1e-5  # 收敛阈值
batch_size = 16  # 每次训练的样本数

t = time.time()  # 记录训练开始时间
print(f"Starting training at {time.ctime(t)}")

# 开始训练循环
for cycle in range(max_cycles):
    optimizer.zero_grad() 
    
    # 前向传播：计算向量v并归一化
    v = model(all_states)
    v_normalized = v / torch.norm(v)
    
    rayleigh_quotient = (v_normalized.T @ H )@ v_normalized
    loss = rayleigh_quotient
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    # 每print_interval个周期打印一次当前的能量值
    if cycle % print_interval == 0:
        print(f"Cycle {cycle}: Energy = {rayleigh_quotient.item():.6f}")
    
    # 检查是否收敛
    if rayleigh_quotient.item() - target_energy < threshold:
        print(f"Converged after {cycle} cycles. Energy: {rayleigh_quotient.item():.6f}")
        break

print(f"Training completed in {time.time() - t:.2f} seconds")  # 打印训练总耗时

# 验证结果
with torch.no_grad():
    v = model(all_states)
    v_normalized = v / torch.norm(v)
    # 计算最终的能量值
    energy = (v_normalized @ H @ v_normalized).item()
    print(f"Final energy: {energy:.6f} (Target: {target_energy})")  # 打印最终能量与目标能量

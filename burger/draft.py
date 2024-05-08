import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置时空网格
h = 0.01
k = 0.01
x = np.arange(-1, 1, h)
t = np.arange(0, 1, k)
X, T = np.meshgrid(x, t)
X = torch.tensor(X.flatten(), dtype=torch.float32)
T = torch.tensor(T.flatten(), dtype=torch.float32)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)
model.eval()

# 计算预测解
XT = torch.stack((X, T), dim=1).to(device)
with torch.no_grad():
    U_pred = model(XT).cpu().numpy().reshape(len(t), len(x))

# 计算精确解，这里需要根据实际问题给出精确解的计算方法
def exact_solution(t, x):
    return np.sin(np.pi * x) * np.exp(-0.01 * np.pi**2 * t)  # 示例：热方程的解

U_exact = exact_solution(T.reshape(len(t), len(x)), X.reshape(len(t), len(x)))

# 计算误差
error = np.abs(U_pred - U_exact)

# 绘制预测解和精确解
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.contourf(x, t, U_pred, levels=50, cmap='jet')
plt.colorbar()
plt.title("Predicted Solution")

plt.subplot(1, 3, 2)
plt.contourf(x, t, U_exact, levels=50, cmap='jet')
plt.colorbar()
plt.title("Exact Solution")

plt.subplot(1, 3, 3)
plt.contourf(x, t, error, levels=50, cmap='jet')
plt.colorbar()
plt.title("Error")

plt.tight_layout()
plt.show()

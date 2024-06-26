import torch
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

# 选择GPU或CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


data = scipy.io.loadmat('./burgers_shock.mat')
Exact = np.real(data['usol']).T



# 从文件加载已经训练完成的模型
model_loaded = torch.load('model.pth', map_location=device)
model_loaded.eval()  # 设置模型为evaluation状态


# 生成时空网格
h = 0.01
k = 0.01
x = torch.arange(-1, 1, h)
t = torch.arange(0, 1, k)
X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
X = X.to(device)

u_exact = -torch.sin(np.pi * X) * torch.exp(-0.01 * np.pi * t)
print("u_exact", u_exact.shape)
# 计算该时空网格对应的预测值
with torch.no_grad():
    U_pred = model_loaded(X).reshape(len(x), len(t)).cpu().numpy()

# 绘制计算结果
plt.rcdefaults()
plt.figure(figsize=(5, 3), dpi=300)

xnumpy = x.numpy()
print("xnumpy", xnumpy.shape)
print("exact", u_exact[:,0])
print("exact", U_pred[:,0].shape)

plt.plot(xnumpy, u_exact[:,0], 'b-', linewidth = 2, label = 'Exact') 
plt.plot(xnumpy, U_pred[:, 0], 'r--', linewidth = 2, label = 'Prediction')

plt.plot(xnumpy, u_exact[:, 20], 'o', markersize=1)
plt.plot(xnumpy, U_pred[:, 40], 'o', markersize=1)
print('u1', U_pred[:,0])
print('u2', U_pred[:,20])
print('u3', U_pred[:,40])
plt.legend()
plt.title("Scatter Plots at Different Time Steps")
plt.xlabel("X Coordinate")
plt.ylabel("Predicted Values")

# Save the first figure
plt.savefig('scatter_plots.png')
plt.close()  # Close the plot to free up memory

# Create the second figure with a heatmap
plt.figure(figsize=(5, 3), dpi=300)
sns.heatmap(U_pred, cmap='jet')
plt.title("Heatmap of Predictions")

# Save the heatmap
plt.figure(figsize=(5, 3), dpi=300)

sns.heatmap(U_pred, cmap='jet')
plt.show()
plt.savefig('heatmap.png')
plt.close()                         # Close the plot to free up memory

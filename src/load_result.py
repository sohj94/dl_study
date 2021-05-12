import sys, os
sys.path.append(os.pardir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

result_prefix = "../data/result/hw1_result_"
dataset = "cifar-10"
result = pd.read_csv(result_prefix + dataset + ".csv")
x = [i for i in range(3,16)]
y = [i for i in range(10,151,10)]

Z = np.array([list(result[str(i)]) for i in range(len(x))]).transpose()
X, Y = np.meshgrid(x, y)
fig = plt.figure(figsize=(9,9))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='black', alpha=0.5)
# ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title(dataset, size=20)
ax.set_xlim([3,15])
ax.set_ylim([10,150])
ax.set_xlabel('# of layers')
ax.set_ylabel('width')

M = np.max(Z)
ax.set_zlim([int(10*M)/10, int(10*M+1)/10])

idx_x = np.argmax(Z)//Z.shape[1]
idx_y = np.argmax(Z)%Z.shape[1]
ax.scatter(x[idx_y], y[idx_x], Z[idx_x, idx_y], color='red', s=100)
ax.text(x[idx_y], y[idx_x], Z[idx_x, idx_y], "  max point: width {} depth {}".format(y[idx_x], x[idx_y]))
plt.savefig(result_prefix + dataset + ".png", dpi=300)

plt.show()
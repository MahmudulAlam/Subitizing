import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['font.family'] = 'lato'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.default'] = 'rm'
plt.rcParams['mathtext.fontset'] = 'cm'

y_true = np.load('data/y_true.npy')
y_pred = np.load('data/y_pred.npy')

n_class = 6
conf_mat = np.zeros((n_class, n_class))
size = y_true.shape[0]

for i in range(size):
    m = int(y_true[i])
    n = int(y_pred[i])
    conf_mat[m, n] += 1

dataset_size = int(6000 * 1.0 / 6)
conf_mat = conf_mat / dataset_size
print(conf_mat)
# np.savetxt('results/resnet/hrr/exp4.csv', conf_mat, delimiter=',')

plt.figure(figsize=[10, 8])
plot = sns.heatmap(conf_mat, vmin=0, vmax=1, annot=True, fmt=".2g", linewidths=2, cmap='OrRd')
plot.set_xticklabels(list(range(1, n_class + 1)))
plot.set_yticklabels(list(range(1, n_class + 1)))
plt.title('')
plt.show()

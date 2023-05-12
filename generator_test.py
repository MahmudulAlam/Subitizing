import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from HRR.with_pytorch import unbinding, cosine_similarity

sns.set_style('darkgrid')
plt.rcParams['font.family'] = 'lato'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 2.0

key = torch.tensor(np.load('data/key.npy'))
value = torch.tensor(np.load('data/value.npy'))
y = torch.tensor(np.load('data/y.npy'))

print(y.shape)
print(y.max(), y.min())

ps_list = []
n = y.shape[0]

for i in reversed(range(n)):
    vi = value[i]
    vi_prime = unbinding(y[i], key[i], dim=-1)

    ps = cosine_similarity(vi, vi_prime)
    ps_list.append(ps)
    # pn = torch.clip(torch.sum(vi * vi_prime), 0, 1)

items = list(range(1, n + 1))
ps_list = list(reversed(ps_list))

blue = {'color': '#4285F4', 'marker': '*', 'markersize': 12}

plt.plot(items, ps_list, linewidth=2.5, label='ps = cosine similarity', **blue)
plt.legend(loc='lower left')

plt.xlim([0, n + 1])
plt.ylim([-0.05, 1.05])

plt.xticks(list(range(1, 7)))
plt.xlabel('# of items')

plt.ylabel('Score')
plt.title('Subitizing')
plt.show()

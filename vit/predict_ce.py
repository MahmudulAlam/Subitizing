import torch
import numpy as np
from dataset import dataset
from network_ce import Network
from utils import index_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

exp = ['0_circle_larger_size_variation', '1_triangle', '2_squares', '4_circle_swapped_color', '5_circle_hollow']

dataset_name = '5_circle_hollow/'

data_loader = dataset(root='../data/',
                      path='Test/' + dataset_name,
                      batch_size=256,
                      split=1.0,
                      augment=False,
                      shuffle=False,
                      num_workers=0)

network = Network()
network.to(device).eval()
network.load_state_dict(torch.load('../weights/vit_ce_circle.h5'))

dataset_size = int(6000 * 1.0)
y_true_out = torch.zeros((dataset_size,))
y_pred_out = torch.zeros((dataset_size,))
index_iter = index_sequence(batch_size=256, dataset_size=dataset_size)


with torch.no_grad():
    for i, data in enumerate(data_loader):
        x_true, y_true = data[0].to(device), data[1].to(device)
        y_pred = network(x_true)
        y_pred = torch.argmax(y_pred, dim=-1)

        a, b = index_iter[i]
        y_true_out[a:b] = y_true.detach().cpu()
        y_pred_out[a:b] = y_pred.detach().cpu()

np.save('../data/y_true.npy', y_true_out)
np.save('../data/y_pred.npy', y_pred_out)

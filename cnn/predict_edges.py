import torch
import numpy as np
from dataset import dataset
from network_hrr import Network
from utils import index_sequence
from HRR.with_pytorch import unbinding, cosine_similarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset_name = '7_random_shape_objects_hollow/'
train_loader, test_loader = dataset(root='../data/',
                                    path='Test/' + dataset_name,
                                    batch_size=256,
                                    split=0.8,
                                    augment=False,
                                    shuffle=False,
                                    num_workers=0)

network = Network()
network.to(device).eval()
network.load_state_dict(torch.load('../weights/edges.h5'))

key = torch.tensor(np.load('../data/key.npy')).to(device)
value = torch.tensor(np.load('../data/value.npy')).to(device)

key = torch.unsqueeze(key, dim=0)
value = torch.unsqueeze(value, dim=0)

dataset_size = int(6000 * 0.2)
y_true_out = torch.zeros((dataset_size,))
y_pred_out = torch.zeros((dataset_size,))
index_iter = index_sequence(batch_size=256, dataset_size=dataset_size)


with torch.no_grad():
    for i, data in enumerate(test_loader):
        x_true, labels = data[0].to(device), data[1].to(device)
        y_pred = network(x_true)
        y_pred = torch.unsqueeze(y_pred, dim=1)
        value_prime = unbinding(y_pred, key, dim=-1)
        score = cosine_similarity(value, value_prime, dim=-1)
        y_pred = torch.argmax(score, dim=-1)

        a, b = index_iter[i]
        y_true_out[a:b] = labels.detach().cpu()
        y_pred_out[a:b] = y_pred.detach().cpu()

np.save('../data/y_true.npy', y_true_out)
np.save('../data/y_pred.npy', y_pred_out)

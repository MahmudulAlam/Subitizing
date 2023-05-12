import time
import torch
import numpy as np
from dataset import dataset
from network_hrr import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

folder = 'circle/'
train_loader = dataset(root='../data/',
                       path='Train/' + folder,
                       batch_size=256,
                       split=1.0,
                       augment=False,
                       shuffle=True,
                       num_workers=0)

network = Network()
network.to(device)
# network.load_state_dict(torch.load('../weights/cnn_hrr_circles.h5'))

epochs = 100
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

y_gt = torch.tensor(np.load('../data/y.npy')).to(device)


def loss_function(true, pred):
    return torch.mean(torch.norm(true - pred))


for epoch in range(1, epochs + 1):
    train_loss = []
    tic = time.time()

    for i, data in enumerate(train_loader, 0):
        x_true, y_true = data[0].to(device), data[1].to(device)
        y_true = y_gt[y_true.long()]
        optimizer.zero_grad()

        y_pred = network(x_true)
        loss = loss_function(y_true, y_pred)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    mean_loss = sum(train_loss) / len(train_loss)
    toc = time.time()

    form = 'Epoch: {0:>3d}/' + str(epochs) + ' || train Loss: {1:>8.6f} || etc: {2:>5.2f}s'
    print(form.format(epoch, mean_loss, toc - tic))

torch.save(network.state_dict(), '../weights/cnn_hrr_circles.h5')
print('All Done!')

import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(Network, self).__init__()

        self.main_network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 7)), activation, nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 32, kernel_size=(5, 5)), activation, nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3)), activation,
            nn.Conv2d(64, 64, kernel_size=(3, 3)), activation,
            nn.Conv2d(64, 128, kernel_size=(3, 3)), activation, nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(6272, 1024), activation,
            nn.Linear(1024, 6), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.main_network(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = torch.normal(0, 1, (32, 1, 100, 100), dtype=torch.float32).to(device)
    network = Network()
    network.to(device)

    outputs = network(inputs)

    print(inputs.shape)
    print(outputs.shape)

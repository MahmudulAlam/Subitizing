import torch
from torch import nn
from vit_pytorch import ViT


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        vit = ViT(image_size=100,
                  patch_size=10,
                  num_classes=64,
                  dim=256,
                  depth=6,
                  heads=4,
                  mlp_dim=512,
                  channels=1,
                  dropout=0.1,
                  emb_dropout=0.1)

        self.main_network = nn.Sequential(
            vit,
            nn.Tanh()
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

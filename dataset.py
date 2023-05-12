import os
import cv2
import torch
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, group, split=1.0, augment=False):
        self.root = root
        self.group = group
        images = os.listdir(root)
        self.length = len(images)

        if group == 'train':
            self.images = images[0:int(self.length * split)]
        else:
            self.images = images[int(self.length * split):]

        if augment:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.RandomAffine(degrees=0,
                                                                         translate=(0, 0),
                                                                         scale=(0.5, 0.5))])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image = cv2.imread(self.root + name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.transform(image)
        label = int(name.split('_')[-1][1:2]) - 1
        label = torch.tensor(label, dtype=torch.int32)
        return image, label


def dataset(root, path, batch_size, split, augment, shuffle, num_workers):
    train_set = Dataset(root=root + path, group='train', split=split, augment=augment)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers)

    if split < 1.0:
        test_set = Dataset(root=root + path, group='test', split=split, augment=augment)

        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)

        return train_loader, test_loader

    else:
        return train_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train, test = dataset(root='./data/',
                          path='Train/circle/',
                          split=0.8,
                          augment=True,
                          batch_size=256,
                          shuffle=True,
                          num_workers=0)

    total = 0
    for x, y in test:
        total += x.shape[0]
        x = x[0].squeeze()
        plt.imshow(x)
        plt.show()
        break
    print(total)

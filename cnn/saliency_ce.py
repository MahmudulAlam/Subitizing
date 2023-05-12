import torch
from dataset import dataset
from network_ce import Network
import matplotlib.pyplot as plt
from utils import index_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

exp = ['0_circle_larger_size_variation/', '1_triangle/', '2_squares/', '4_circle_swapped_color/', '5_circle_hollow/']
counter = 0

for dataset_name in exp:
    true_counter = 0
    false_counter = 0
    data_loader = dataset(root='../data/',
                          path='Test/' + dataset_name,
                          batch_size=1,
                          split=1.0,
                          augment=False,
                          shuffle=False,
                          num_workers=0)

    network = Network()
    network.to(device).eval()
    network.load_state_dict(torch.load('../weights/cnn_ce_circles.h5'))

    dataset_size = int(6000 * 1.0)
    index_iter = index_sequence(batch_size=256, dataset_size=dataset_size)


    def saliency(image, label, model):
        global counter
        global true_counter
        global false_counter

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        input_image = image[0].detach().cpu().numpy()

        image.requires_grad = True
        pred = model(image)
        correct = torch.argmax(pred, dim=-1) == label
        correct = correct[0].detach().cpu()

        if correct and true_counter >= 10:
            return
        if not correct and false_counter >= 10:
            return

        score, indices = torch.max(pred, 1)
        score.backward()

        slc, _ = torch.max(torch.abs(image.grad[0]), dim=0)
        slc = (slc - slc.min()) / (slc.max() - slc.min())
        if correct:
            true_counter += 1
            plt.imsave(f'../saliency/ce/true/{counter}_img.jpg', input_image[0], cmap='gray')
            plt.imsave(f'../saliency/ce/true/{counter}_slc.jpg', slc.detach().cpu().numpy(), cmap='turbo')
        else:
            false_counter += 1
            plt.imsave(f'../saliency/ce/false/{counter}_img.jpg', input_image[0], cmap='gray')
            plt.imsave(f'../saliency/ce/false/{counter}_slc.jpg', slc.detach().cpu().numpy(), cmap='turbo')
        counter += 1
        return


    for i, data in enumerate(data_loader):
        x_true, y_true = data[0].to(device), data[1].to(device)
        saliency(x_true, y_true, network)

        if i == 1024:
            break

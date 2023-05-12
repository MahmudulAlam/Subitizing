import torch


def index_sequence(batch_size, dataset_size):
    index_i = list(range(0, dataset_size, batch_size))
    index_j = list(range(batch_size, dataset_size, batch_size))
    index_j.append(dataset_size)
    return list(zip(index_i, index_j))


def normalize(x):
    max_, _ = torch.max(x, dim=-1, keepdim=True)
    min_, _ = torch.min(x, dim=-1, keepdim=True)
    return (x - min_) / (max_ - min_)


def loss_function(y_true, y_pred, eps=1e-10):
    y_t = torch.clip(y_true, eps, 1.)
    y_p = torch.clip(y_pred, eps, 1.)
    y_t_prime = torch.clip(1. - y_true, eps, 1.)
    y_p_prime = torch.clip(1. - y_pred, eps, 1.)
    bce = - y_t * torch.log(y_p) - y_t_prime * torch.log(y_p_prime)
    loss = torch.sum(bce, dim=-1)
    return torch.mean(loss)

import torch
import torchvision
from torch.utils.data import DataLoader


def official_set(data_name, batch_size, is_download=False, is_dataloader=False):
    if data_name is 'MNIST':
        train_data = torchvision.datasets.MNIST(
            root='./data/mnist/',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=is_download
        )
        train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True
        )
        test_data = torchvision.datasets.MNIST(
            root='./data/mnist/',
            train=False,
            transform = torchvision.transforms.ToTensor()
        )
        test_dataloader = DataLoader(
            dataset=test_data,
            batch_size=1,
            shuffle=False
        )
    if is_dataloader:
        return train_dataloader, test_dataloader
    else:
        return train_data, test_data

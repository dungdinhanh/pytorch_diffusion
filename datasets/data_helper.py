import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os


def _data_transforms_cifar10():
    """Get data transforms for cifar10."""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transform, valid_transform


def get_cifar_loader(train=True, test=False, batch_size=64):

    train_transform, test_transform = _data_transforms_cifar10()
    if train:
        train_set = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        train_loader = None

    if test:
        test_set = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        test_loader = None

    return train_loader, test_loader
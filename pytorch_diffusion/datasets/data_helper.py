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


def get_cifar_loader(train=True, test=False):

    train_transform, test_transform = _data_transforms_cifar10()
    if train:
        train_loader = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    else:
        train_loader = None

    if test:
        test_loader = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
    else:
        test_loader = None

    return train_loader, test_loader
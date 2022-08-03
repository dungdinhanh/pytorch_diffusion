import torchvision.datasets as datasets
import os

cifar_testset = datasets.CIFAR10(root='./temp_data', train=False, download=True, transform=None)

folder = "temp_data/cifar10_test"

os.makedirs(folder, exist_ok=True)
for i in range(len(cifar_testset)):
    img, target = cifar_testset[i]
    img.save(os.path.join(folder, "IMG{:06}.png".format(i)))

folder = "temp_data/cifar10_train"

os.makedirs(folder, exist_ok=True)
for i in range(len(cifar_testset)):
    img, target = cifar_testset[i]
    img.save(os.path.join(folder, "IMG{:06}.png".format(i)))
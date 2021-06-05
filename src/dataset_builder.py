import sys, os
sys.path.append(os.pardir)

import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms

def load_data_set(data_name) :
    if data_name == 'mnist' :
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))
            ])
        train_dataset = MNIST(root='../data/mnist', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='../data/mnist', train=False, download=True, transform=transform)

    if data_name == 'cifar-10' :
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        train_dataset = CIFAR10(root='../data/cifar-10', train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root='../data/cifar-10', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

if __name__ == "__main__" :
    mnist_train, _ = load_data_set('mnist')
    cifar_train, _ = load_data_set('cifar-10')

    print(torch.numel(mnist_train[0][0]))
    print(cifar_train[0][0].shape)
    print(cifar_train[0])
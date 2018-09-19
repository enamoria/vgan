# -*- coding: utf-8 -*-

""" Created on 2:56 PM 9/18/18
    @author: ngunhuconchocon
    @brief:
"""

from __future__ import print_function

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from utils import Logger, Generator, Discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from IPython import display
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


batch_size = 4
nb_epoch = 200


def mnist_data():
    print("MNIST data ...")
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    out_dir = os.path.join(os.path.dirname(__file__), "dataset")
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=False)


def noise(size):
    return torch.rand(size, 100)


def train_discriminator(optimizer, real_data, fake_data):
    # Train on real ...
    # zero_grad
    optimizer.zero_grad()

    # calculate output of the net
    D_real_pred = D(real_data)

    error_real = loss(D_real_pred, torch.ones(real_data.size(0), 1))
    error_real.backward()

    # Train on fake
    D_fake_pred = D(fake_data)
    error_fake = loss(D_fake_pred, torch.zeros(fake_data.size(0), 1))
    error_fake.backward()

    # optimize the optimizer
    optimizer.step()

    return error_real + error_fake, D_real_pred, D_fake_pred


def train_generator(optimizer, fake_data):
    optimizer.zero_grad()

    D_decision_on_fake = D(fake_data)
    error_generator = loss(D_decision_on_fake, torch.ones(fake_data.size(0)))
    error_generator.backward()

    optimizer.step()

    return error_generator


def images_to_vectors(images):
    return images.view(images.size(0), 28*28)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


if __name__ == "__main__":
    # mnist_data()
    data = mnist_data()
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # xxx = iter(train_loader).__next__()[0].numpy()
    # print(np.unique(xxx))

    nb_batch = len(train_loader)
    noise(100)

    D = Discriminator()
    G = Generator()

    d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002)

    loss = nn.BCELoss()

    d_steps = 1

    num_test_sample = 16
    test_noise = noise(num_test_sample)
    logger = Logger(model_name='VGAN', data_name='MNIST')

    for epoch in range(nb_epoch):
        print("Epoch {}".format(epoch))
        for i, data_train in enumerate(train_loader):
            # print(data_train[0].shape, data_train[1].shape)

            # 1: Train discriminator
            real_data = images_to_vectors(data_train[0])
            fake_data = G(noise(real_data.size(0))).detach()

            # Train D
            d_error, d_real_pred, d_fake_pred = train_discriminator(d_optimizer, real_data, fake_data)

            # 2: Train generator
            fake_data = G(noise(real_data.size(0))).detach()
            g_error = train_generator(g_optimizer, fake_data)

            logger.log(d_error, g_error, epoch, i, nb_batch)

            if i % 100 == 0:
                display.clear_output(True)

                test_images = vectors_to_images(G(test_noise)).data.cpu()

                logger.log_images(test_images, num_test_sample, epoch, i, nb_batch)

            logger.save_models(G, D, epoch)

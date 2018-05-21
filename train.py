import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from numpy import prod
from datetime import datetime
from model import CapsuleNetwork
from loss import CapsuleLoss
from time import time
from capsule_net import CapsuleNet, CapsuleLoss
import argparse
from tqdm import tqdm


def train(model, classes, epochs, criterion, optimizer, scheduler, trainloader, device):
    one_hot = torch.eye(len(classes)).to(device)
    for epoch in range(epochs-1):

        running_loss = 0.0
        correct = 0
        total = 0
        for batch_i, (images, labels) in tqdm(enumerate(trainloader, 0)):
            images, labels = images.to(device), labels.to(device)
            labels = one_hot[labels]

            optimizer.zero_grad()

            outputs, reconstructions = model(images)
            loss = criterion(outputs, labels, images, reconstructions)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            accuracy = float(correct) / float(total)

            print('Epoch: {}, Batch: {}, Loss: {}, Accuracy: {}'
                .format(epoch, batch_i+1, running_loss/(batch_i+1), accuracy))

        scheduler.step()


def main():
    SAVE_MODEL_PATH = 'trained/'

    if not os.path.exists(SAVE_MODEL_PATH):
        os.mkdir(SAVE_MODEL_PATH)

    DATA_PATH = '.data/'

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('dataset', nargs='?', type=str, default='MNIST',
                        help="'MNIST' or 'CIFAR' (case insensitive).")
    # whether or not to use GPU
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        choices=['cuda', 'cpu'], help='Device to use. Choose "cuda" for GPU or "cpu".')
    # batch size
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size, default is 16')

    # traning epoch
    parser.add_argument('--epoch', type=int, default=50, help='training epoch')

    # primary capsule
    parser.add_argument('--dim_caps', type=int, default=8, help='dimension of each capsule, default is 8')
    # out capsule
    parser.add_argument('--out_caps', type=int, default=16)
    # conv in_channels
    parser.add_argument('--in_conv_channels', type=int, default=1)
    # conv out_channels
    parser.add_argument('--out_conv_cahnnels', type=int, default=256)
    # num of routing
    parser.add_argument('--num_routing', type=int, default=3)
    # lr
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    # Exponential learning rate decay
    parser.add_argument('--lr_decay', type=float, default=0.96,
                    help='Exponential learning rate decay.')
    # data path
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    # channels in primary capsule layer
    parser.add_argument('--primary_channels', type=int, default=32, help='num of Channels in PrimaryCapsule layer')

    args = parser.parse_args()

    # GPU or CPU
    device = torch.device(args.device)

    assert (args.dataset.upper() == 'MNIST' or args.dataset.upper() == 'CIFAR'), 'dataset must be MNIST or CIFAR'

    print('===> Data loading')

    if args.dataset.upper() == 'MNIST':
        args.data_path = os.path.join(args.data_path, 'MNIST')
        size = 28
        classes = list(range(10))
        mean, std = ((0.1307,), (0.3081,))
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        trainset = torchvision.datasets.MNIST(root=args.data_path, train=True,
                                                    download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root=args.data_path, train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=2)
        in_conv_channels = 1
        out_conv_cahnnels = 256
    elif args.dataset.upper() == 'CIFAR':
        args.data_path = os.path.join(args.data_path, 'CIFAR')
        size = 32
        classes = ['plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
        mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=2)

        in_conv_channels = 3
        out_conv_cahnnels = 256

    print('===> Data loaded')
    print('===> Building model')
    img_shape = trainloader.dataset[0][0].numpy().shape
    model = CapsuleNet(img_shape, in_conv_channels, out_conv_cahnnels, 
                       args.primary_channels, args.dim_caps, len(classes), args.out_caps, args.num_routing, 
                       device)
    # set model to device [CPU or GPU]
    model = model.to(device)
    # Are we using GPU
    print('\nDeivce: {}'.format(device))

    # Print model architecture and parameters
    print('Model architectures:\n{}\n'.format(model))
    print('Parameters and size:')
    for name, param in model.named_parameters():
        print('{}: {}'.format(name, list(param.size())))

    criterion = CapsuleLoss()
    learning_rate = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    print('===> Training')
    train(model, classes, args.epoch,
         criterion, optimizer, scheduler, trainloader,
         device)

    print('===> Testing')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            outputs, reconstructions = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on {}: {}'.format(args.dataset.upper(), 100*correct/total))


if __name__ == '__main__':
    main()

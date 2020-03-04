# -*- coding: UTF-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import *
import torchvision
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import utils
from tensorboardX import SummaryWriter

num_epoches = 120
batch_size = 64
num_classes = 65
lr = 0.01
step_size = 30

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

base_path = '/workspace/fubo/'
train_path = base_path + 'train/'
valid_path = base_path + 'valid/'

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

valid_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset_train = datasets.ImageFolder(root=train_path, transform=train_transform)
dataset_valid = datasets.ImageFolder(root=valid_path, transform=valid_transform)

train_loader = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

valid_loader = torch.utils.data.DataLoader(
    dataset=dataset_valid,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=0, std=0.02)
        init.constant_(m.bias.data, 0)
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        init.normal_(m.bias.data, std=0.02)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, pool=None):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.pool = pool

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)

        return x


class Net(nn.Module):
    def __init__(self, num_class):
        super(Net, self).__init__()

        # self.feature = nn.Sequential(
        #     BasicBlock(3, 8),
        #     BasicBlock(8, 16, nn.MaxPool2d(2)),
        #     BasicBlock(16, 32),
        #     BasicBlock(32, 48, nn.MaxPool2d(2)),
        #     BasicBlock(48, 64),
        #     BasicBlock(64, 128, nn.MaxPool2d(2)),
        #     BasicBlock(128, 256),
        #     BasicBlock(256, 256, nn.MaxPool2d(2)),
        #     BasicBlock(256, 512),
        #     BasicBlock(512, 1024, nn.MaxPool2d(2)),
        # )

        self.feature = nn.Sequential(
            BasicBlock(3, 8),
            BasicBlock(8, 16, nn.MaxPool2d(2)),
            BasicBlock(16, 24),
            BasicBlock(24, 32, nn.MaxPool2d(2)),
            BasicBlock(32, 48),
            BasicBlock(48, 64, nn.MaxPool2d(2)),
            BasicBlock(64, 72),
            BasicBlock(72, 128, nn.MaxPool2d(2)),
            BasicBlock(128, 256),
            BasicBlock(256, 256, nn.MaxPool2d(2)),
        )

        self.out = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_class),
        )

        self.apply(weight_init)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


if __name__ == '__main__':
    model = Net(num_classes).to(device)

    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)
    loss_func = torch.nn.CrossEntropyLoss()

    summary_writer = SummaryWriter()
    dump_input = torch.rand(1, 3, 224, 224).to(device)
    summary_writer.add_graph(model, (dump_input,), verbose=False)

    for epoch in range(num_epoches):

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader, start=1):  # 每一步 loader 释放一小批数据用来学习
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            running_loss += loss.data.item() * batch_y.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == batch_y).sum()
            running_acc += num_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Finish {} epoch, Loss: {:.6f}, Accuracy of Train: {:.6f}'.format(
            epoch + 1, running_loss / (len(dataset_train)), running_acc / (len(dataset_train))))

        scheduler.step()
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        summary_writer.add_scalar('train_loss', running_loss / (len(dataset_train)), epoch)
        summary_writer.add_scalar('train_acc', running_acc / (len(dataset_train)), epoch)

        model.eval()
        correct = 0
        total = 0
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        summary_writer.add_scalar('top1', accuracy, epoch)
        print('Accuracy of Test: %.2f %%' % (accuracy))

    torch.save(model, "netC.pth")

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

num_epoches = 20
batch_size = 64
num_classes = 65
lr = 0.0001
step_size = 10

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

base_path = '/workspace/fubo/'
train_path = base_path + 'train/'
valid_path = base_path + 'valid/'

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
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


class ResNet50Fe(nn.Module):
    def __init__(self):
        super(ResNet50Fe, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = ResNet50Fe()
        self.classifier = nn.Linear(self.features.output_num(), num_classes)

        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                init.normal_(m.bias.data, std=0.02)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    resnet50 = ResNet50().to(device)

    optimizer = torch.optim.Adam(resnet50.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    loss_func = torch.nn.CrossEntropyLoss()

    summary_writer = SummaryWriter()
    dump_input = torch.rand(1, 3, 224, 224).to(device)
    summary_writer.add_graph(resnet50, (dump_input,), verbose=False)

    for epoch in range(num_epoches):

        resnet50.train()
        running_loss = 0.0
        running_acc = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader):  # 每一步 loader 释放一小批数据用来学习
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            out = resnet50(batch_x)
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

        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        summary_writer.add_scalar('train_loss', loss.item(), epoch)

        resnet50.eval()
        correct = 0
        total = 0
        for data in valid_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = resnet50(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        summary_writer.add_scalar('top1', accuracy, epoch)
        print('Accuracy of Test: %.2f %%' % (accuracy))

    torch.save(resnet50, "resnet50_pretrain.pth")

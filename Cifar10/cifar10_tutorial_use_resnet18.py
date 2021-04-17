import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 加载数据
def get_data():
    transform = transforms.Compose([transforms.ToTensor()
                                    # transforms.Resize((32, 32)),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    trainset = torchvision.datasets.CIFAR10(root='cifar_data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='cifar_data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=True)

    return trainloader, testloader


# 显示图片
def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


# 定义卷积神经网络
class Net(nn.Module):
    def __init__(self, batch_size, height, width):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 残差结构
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(
            ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(
            ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.extra = nn.Sequential(nn.BatchNorm2d(ch_out),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out),
                nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        out = F.relu(self.pool(self.bn1(self.conv1(x))))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.extra(x) + out
        return out


# ResNet网络
class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )
        self.blk1 = ResBlk(16, 16)
        self.blk2 = ResBlk(16, 32)
        self.outlayer = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 32 * 8 * 8)
        x = self.outlayer(x)
        return x


def train_net(net, trainloader, testloader):

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print('Start Training!')
    for epoch in range(3):
        for batch_idx, data in enumerate(trainloader, 0):

            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
        correct_num = 0
        for inputs, labels in testloader:
            outputs = net(inputs)
            predicted = outputs.argmax(dim=1)
            correct_num += torch.eq(predicted, labels).float().sum().item()
        print('Test set Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct_num, len(testloader.dataset),
            100. * correct_num / len(testloader.dataset)))
    print('Finished Training!')


def main():
    # 加载数据
    trainloader, testloader = get_data()
    # 构建网络
    net = ResNet()
    print(net)
    train_net(net, trainloader, testloader)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(32)))

    outputs = net(images)
    predicted = outputs.argmax(dim=1)
    print('Predicted: ', ' '.join('%5s' %
                                  classes[predicted[j]] for j in range(32)))


if __name__ == '__main__':
    main()

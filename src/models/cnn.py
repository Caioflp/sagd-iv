import torch
import torch.nn as nn
import torch.nn.functional as F


class DefaultCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.LazyLinear(32)
        self.fc3 = nn.LazyLinear(1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.LazyLinear(32)
        self.fc3 = nn.LazyLinear(1)

    def forward(self, xz):
        x, z = xz[:, 0:1], xz[:, 1:].view(-1, 1, 28, 28)
        z = self.pool(F.relu(self.conv1(z)))
        z = self.pool(F.relu(self.conv2(z)))
        z = torch.flatten(z, 1) # flatten all dimensions except batch
        xz = torch.concat([x, z], axis=1)
        xz = F.relu(self.fc1(xz))
        xz = F.relu(self.fc2(xz))
        xz = self.fc3(xz)
        return xz

if __name__ == "__main__":
    x = torch.ones((3, 1))
    z = torch.ones((3, 28*28))
    xz = torch.ones((3, 28*28+1))
    model_1 = ResidualCNN()
    model_2 = DefaultCNN()
    print(model_1(xz))
    print(model_2(z))

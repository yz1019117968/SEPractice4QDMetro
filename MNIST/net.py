import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 实例化Conv2d网络为conv1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        # [64, 1, 28, 28]
        x = self.conv1(x)
        # [64, 1, 28-5+1, 28-5+1] = [64, 1, 24, 24]
        x = F.relu(F.max_pool2d(x, 2))
        # [64, 1, 24/2, 24/2] = [64, 10, 12, 12]
        x = self.conv2(x)
        # [64, 20, 8, 8]
        x = self.conv2_drop(x)
        # [64, 20, 8, 8]
        x = F.relu(F.max_pool2d(x, 2))
        # [64, 20, 4, 4]
        # 指定最后一个维度为320个元素
        x = x.view(-1, 320)
        # [64, 320]
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # [64, 10]
        x = self.fc2(x)
        # return log(softmax(x))
        return F.log_softmax(x)
import torch
import torch.nn as nn
import torch.nn.functional as F





class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size = 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 2)
        self.fc1 = nn.Linear(128, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 2)
         
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
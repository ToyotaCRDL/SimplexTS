import torch.nn as nn
import torch.nn.functional as F


        
class Branch(nn.Module):
    def __init__(self, input):
        super(Branch, self).__init__()
        self.branch1 = nn.Linear(input, 1024)
        self.branch2 = nn.Linear(1024, 1024)
        self.branch3 = nn.Linear(1024, 512)
        self.branch4 = nn.Linear(512, 256)
        self.branch5 = nn.Linear(256, 1)


    def forward(self, x):
        x = F.relu(self.branch1(x))
        x = F.relu(self.branch2(x))
        x = F.relu(self.branch3(x))
        x = F.relu(self.branch4(x))
        x = self.branch5(x)

        return x

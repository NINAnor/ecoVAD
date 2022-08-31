import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG11(nn.Module):
    def __init__(self):
        super().__init__()

        # First set of conv layers -> depth of 64
        self.conv11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn11  = nn.BatchNorm2d(64)
        
        # Second set of conv layers -> from depth 64 to depth 128
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21  = nn.BatchNorm2d(128)
        
        # Third set of conv layers -> from depth 128 to depth 256
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31  = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32  = nn.BatchNorm2d(256)
                      
        # Fourth set of conv layers -> from depth 128 to depth 256
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41  = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42  = nn.BatchNorm2d(512)
        
        # Fifth set of conv layers -> from depth 128 to depth 256
        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51  = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52  = nn.BatchNorm2d(512)
              
        # First FC layer
        self.fc1 = nn.Linear(4 * 4 * 512,  4096)
        # Second FC layer
        self.fc2 = nn.Linear( 4096,  4096)
        
        # Add a dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
        # Output
        self.fc3 = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # MaxPool for the first block --> img from 128x128 to 64x64
        out = F.max_pool2d(torch.relu(self.bn11(self.conv11(x))), 2)

        # MaxPool for the first block --> img from 64x64 to 32x32
        out = F.max_pool2d(torch.relu(self.bn21(self.conv21(out))), 2)

        # MaxPool for the first block --> img from 32x32 to 16x16
        out = torch.relu(self.bn31(self.conv31(out)))
        out = F.max_pool2d(torch.relu(self.bn32(self.conv32(out))), 2)
        
        # MaxPool for the first block --> img from 16x16 to 8x8
        out = torch.relu(self.bn41(self.conv41(out)))
        out = F.max_pool2d(torch.relu(self.bn42(self.conv42(out))), 2)
        
        # MaxPool for the first block --> img from 8x8 to 4x4
        out = torch.relu(self.bn51(self.conv51(out)))
        out = F.max_pool2d(torch.relu(self.bn52(self.conv52(out))), 2)
        
        # Flatten the whole thing: image of 4 x 4 * 512 
        out = out.view(-1, 4 * 4 * 512)
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.dropout(torch.relu(self.fc2(out)))
        
        # Sigmoid included in the loss function
        out = self.sigmoid(self.fc3(out))
        return(out)
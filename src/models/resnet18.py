import torch
import torchvision

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class ResNet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = torchvision.models.resnet18(pretrained=False)
        self.feat_extract.fc = Identity()
        self.output_size = 512
    
    def forward(self, x):
        return self.feat_extract(x)
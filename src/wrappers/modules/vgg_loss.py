import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) 
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, return_features=False):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        input = ((input+1)/2-self.mean) / self.std
        target = ((target+1)/2-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            with torch.no_grad():
                y = block(y)
            loss += torch.nn.functional.l1_loss(x, y.detach())
        if return_features:
            return loss, x, y
        else:
            return loss

    def get_features(self, input):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        
        input = ((input+1)/2-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats

    def compare_features(self, input, target):
        loss = 0.0
        for x, y in zip(input, target):
            loss += torch.nn.functional.l1_loss(x, y.detach())
        return loss
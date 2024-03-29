import torchvision
import torch.nn as nn

__all__ = ['ResNet18', 'ResNet50']


class ResNet18(nn.Module):
    output_size = 512

    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=pretrained)
        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool','fc']:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        pool_f=pool.reshape(pool.size(0),-1)
        out = self.fc(pool_f)

        if get_ha:
            return b1, b2, b3, b4, pool,out

        return pool,out


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool','fc']:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        pool_f = pool.reshape(pool.size(0), -1)
        out = self.fc(pool_f)
        if get_ha:
                return b1, b2, b3, b4, pool,out
        return pool,out
class ResNet152(nn.Module):
    output_size = 2048
    def __init__(self, pretrained = True):
        super(ResNet152, self).__init__()
        model = torchvision.models.resnet152(pretrained=pretrained)
        #print(model)
        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool','fc']:
            self.add_module(module_name,getattr(model, module_name))
    def forward(self,x, get_ha=False):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            b1 = self.layer1(x)
            b2 = self.layer2(b1)
            b3 = self.layer3(b2)
            b4 = self.layer4(b3)
            pool = self.avgpool(b4)
            pool = pool.reshape(pool.size(0), -1)
            out = self.fc(pool)
            if get_ha:
                return b1, b2, b3, b4, pool,out
            return pool, out

import torchvision
import torch.nn as nn

__all__ = ['ShuffleNet']


class ShuffleNet(nn.Module):
    output_size = 2048

    def __init__(self, pretrained=True):
        super(ShuffleNet, self).__init__()
        base_model= torchvision.models.shufflenet_v2_x2_0(pretrained=pretrained)
        #print(base_model)
        for module_name in ['conv1','maxpool', 'stage2', 'stage3', 'stage4', 'conv5', 'fc']:
            self.add_module(module_name,getattr(base_model, module_name))
    def forward(self,x, get_ha=False):
            s1= self.maxpool(self.conv1(x))
            s2 = self.stage2(s1)
            s3 = self.stage3(s2)
            s4 = self.stage4(s3)
            c5 = self.conv5(s4)
            c5_f = c5.mean([2, 3])
            output = self.fc(c5_f)
            if get_ha:
                return s1, s2, s3, s4, c5,output
            return c5,output
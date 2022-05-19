from operator import mod
import torch
import torch.nn as nn
import torchvision


C0 = []
C1 = []
C2 = []
C3 = []
C4 = []



class Resnet101Backbone(nn.Module):
    def __init__(self):
        super(Resnet101Backbone, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        C0 = [] #entry layers
        for i, (name, module) in enumerate(model.named_children()):
            if i < 4:
                C0.append(module)
            if name == 'layer1':
                self.C1 = module
            if name == 'layer2':
                self.C2 = module
            if name == 'layer3':
                self.C3 = module
            if name == 'layer4':
                self.C4 = module
            
            self.C0 = nn.Sequential(*C0)
    
    def forward(self, x):
        x = self.C0(x)
        x = self.C1(x)
        C2 = self.C2(x)
        C3 = self.C3(C2)
        C4 = self.C4(C3)
        #return [C2, C3, C4]
        #TODO: SET BACKBONE TO EVAL() TO FREEZE CHECK BEST METHOD IF NOT EVAL()

x = torch.randn(1, 3, 224, 224)
model = Resnet101Backbone()
x = model(x)
for i in x:
    print(i.shape)
layers = [module for module in model.modules()][:-2
model = nn.Sequential(*[module for name, module in model.modules() if name not in ['avgpool', 'fc']])


# Access names of layers in model
for name, module in model.named_children():
    print(name)

layers = [module for name, module in model.named_children() if name not in ['avgpool', 'fc']]
layers
# collect layers by layer type
layers = [module for module in model.modules() if isinstance(module, nn.Sequential)]


for layer in layers:
    print(layer)
    input('press to continue')
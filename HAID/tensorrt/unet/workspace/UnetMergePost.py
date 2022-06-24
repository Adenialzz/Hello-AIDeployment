import torch
from nets.unet import Unet

class MyUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = Unet(num_classes=21, backbone='vgg')
        state_dict = torch.load("./1.5-unet/unet_voc.pth", map_location='cpu')
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        y = self.model(x)
        y = y.permute(0, 2, 3, 1).softmax(dim=-1)
        return y

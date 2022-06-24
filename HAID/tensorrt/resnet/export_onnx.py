import torch
import torchvision.models as models

# model = models.resnet50(pretrained=True)

class ResNet50_wSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y = self.base_model(x)
        prob = self.softmax(y)
        return prob

if __name__ == '__main__':
    # model = models.resnet50(pretrained=True)
    model = ResNet50_wSoftmax()   # 将后处理添加到模型中
    dummpy_input = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
            model, dummpy_input, 'resnet50_wSoftmax.onnx',
            input_names=['image'],
            output_names=['predict'],
            opset_version=11,
            dynamic_axes={'image': {0: 'batch'}, 'predict': {0: 'batch'}}
    )

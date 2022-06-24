import torch
import torchvision.models as models
import cv2
import numpy as np
from export_onnx import ResNet50_wSoftmax

# model = models.resnet50(pretrained=True)
model = ResNet50_wSoftmax()
model.eval()

image_path = './workspace/125.jpg'
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
img = cv2.imread(image_path)[:, :, ::-1]
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = (img - imagenet_mean) / imagenet_std
img = img.transpose(2, 0, 1).astype(np.float32)
tensor_img = torch.from_numpy(img)[None]

pred = model(tensor_img)[0]
max_idx = torch.argmax(pred)
print(f"max_idx: {max_idx}, max_logit: {pred[max_idx].item()})")


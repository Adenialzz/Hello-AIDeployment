import torch
import cv2
import numpy as np
from nets.unet import Unet

device = 'cpu'
colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                (128, 64, 12)]


ifMergePost = True
if not ifMergePost:
    model = Unet(num_classes=21, backbone='vgg')
    state_dict = torch.load('../unet-pytorch/model_data/unet_vgg_voc.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
else:
    from UnetMergePost import MyUnet
    model = MyUnet().to(device)
model = model.to(device)

image = cv2.imread('../unet-pytorch/img/street.jpg')
image = cv2.resize(image, (512, 512))

# To RGB
image = image[..., ::-1]

# preprocess
image = (image / 255.0).astype(np.float32)

# to tensor
image = image.transpose(2, 0, 1)[None]

image = torch.from_numpy(image).to(device)

with torch.no_grad():
    if ifMergePost:
        prob = model(image)
    else:
        predict = model(image)

# postprocess
if not ifMergePost:
    prob = torch.nn.functional.softmax(predict[0].permute(1, 2, 0), dim=-1).cpu()

label_map = prob.argmax(dim=-1)

seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(label_map, [-1])], [512, 512, -1])
cv2.imwrite('res.jpg', seg_img)



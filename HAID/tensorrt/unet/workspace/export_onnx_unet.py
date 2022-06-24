import torch
from nets.unet import Unet
from UnetMergePost import MyUnet

ifMergePost = True

if ifMergePost:
    model = MyUnet()
else:
    model = Unet()
    ckpt_path = './1.5-unet/unet_voc.pth'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)

dummy = torch.zeros(1, 3, 512, 512)

output_name = 'probablity' if ifMergePost else 'predict'
torch.onnx.export(
        model, (dummy, ), 'unet.onnx', 
        input_names=['image'],
        output_names=[output_name, ],
        opset_version=11,
        dynamic_axes={'image': {0: 'batch'}, output_name: {0: 'batch'}}
    )

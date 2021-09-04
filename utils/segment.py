import sys
sys.path.append('../')

import numpy as np
import torch
from torchvision import transforms

from models.segmentation_models.model import BiSeNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def segment_image(image,modelpath = './pretrained_models/79999_iter.pth', size = 256):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(modelpath, map_location=DEVICE) )
    net.to(DEVICE)
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
    
    #size = 256,256
    with torch.no_grad():
      img = to_tensor(image)
      img = torch.unsqueeze(img, 0)
      img = img.to(DEVICE)
      out = net(img)[0]
      img = inv_normalize(img)
      output = (np.transpose(np.array(out.squeeze(0).cpu()),(1,2,0)).argmax(2).astype(np.uint8))
    return output

def return_hair_mask(image):
  out = segment_image(image)
  hairmask =  np.where(out==17,1,0)  #index corresponding to hair segmentation
  return hairmask





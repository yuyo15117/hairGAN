import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


from models.hypergan_models.model import Model
from .segment import return_hair_mask


__all__ = ['inpainting']


def inpainting(images_path,generator,savepath = None,size=256):

  balds= []

  #filenames = random.sample(os.listdir(dspth), samples)
  filenames = os.listdir(images_path)

  for fname in filenames:

    gt_image = plt.imread(os.path.join(images_path,fname))
    gt_image = cv2.resize(gt_image , (size,size))/255
    gt_image = np.array(gt_image, dtype = np.float32)


    mask = return_hair_mask(gt_image)
    mask = np.array(mask)
    gt_image = np.expand_dims(gt_image, axis=0)


    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(np.array(mask , dtype= np.uint8), kernel, iterations=2)
    if mask.max()>1:
      mask = mask/255
    mask = mask[None,...,None]


    input_image = np.where(mask == 1, 1, gt_image)
    input_image = input_image[0][None,...]


    prediction_coarse, prediction_refine = generator([input_image, mask], training=False)

    normalized = (np.array(prediction_refine[0])- np.array(prediction_refine[0]).min())/ \
    ((np.array(prediction_refine[0]).max())-(np.array(prediction_refine[0]).min()))
    balds.append(prediction_refine)

    if savepath is not None:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        plt.imsave(os.path.join(savepath,fname) , normalized)

  return balds

    
import os
import sys
sys.path.append('../')

import io
import numpy as np
from PIL import Image
import cv2
import IPython.display
from natsort import natsorted
import matplotlib.pyplot as plt
import face_alignment

from .inverter import StyleGANInverter
from models.invert_model.helper import build_generator
from .face_landmark_detector import FaceLandmarkDetector




# def align_face(image_path, align_size=256):
#   """Aligns a given face."""
#   model = FaceLandmarkDetector(align_size)
#   face_infos = model.detect(image_path)
#   imgs = []
#   if len(face_infos) != 0:
#     face_infos = face_infos
#     for info in face_infos:
#       imgs.append(model.align(info))

#   else:
#     return None
#   return imgs


def align_face(image_path, align_size=256 , use_hog = False):
  """Aligns a given face."""
  model = FaceLandmarkDetector(align_size , use_hog= use_hog)
  face_infos = model.detect(image_path)

  aligned_images = []
  aligned_images_names = []


  if len(face_infos) != 0:

    if len(face_infos) == 1:
      info = face_infos[0]
      aligned_image = model.align(info)
      aligned_image_name = info['image_path'].split('/')[-1].split('.')[0]+".jpg"
      return aligned_image , aligned_image_name

    elif len(face_infos) > 1:
      for b,info in enumerate(face_infos):
        aligned_image = model.align(info)
        aligned_image_name = info['image_path'].split('/')[-1].split('.')[0]+f'_{b}.jpg'
        
        aligned_images.append(aligned_image)
        aligned_images_names.append(aligned_image_name)

      return aligned_images,aligned_images_names
  else:
    return None, None

def build_inverter(model_name,pretrained_weights,logger = None, iteration=300, regularization_loss_weight=0.6 , loss_weight_ssim=3 ):
  """Builds inverter"""
  inverter = StyleGANInverter(
      model_name,
      pretrained_weights = pretrained_weights,
      learning_rate=0.01,
      iteration=iteration,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=5e-5,
      regularization_loss_weight=regularization_loss_weight,
      loss_weight_ssim = loss_weight_ssim,
      logger = logger,
      
      )
  return inverter



def get_generator(model_name):
  """Gets model by name"""
  return build_generator(model_name)


def align(inverter, image_path):
  """Aligns an unloaded image."""
  aligned_images , aligned_images_names  = align_face(image_path,
                             align_size=256)
  return aligned_images , aligned_images_names


def invert(inverter, image):
  """Inverts an image which has been preprocessed."""
  latent_code, reconstruction , ssim_loss = inverter.easy_invert(image, num_viz=1)
  return latent_code, reconstruction, ssim_loss


def batch_invert(inverter,source_dir,threshold = 0.3,need_align = False):
  """Inverts a directory of images which has not been preprocessed.
     cropping and alignment done inside this function.
  """

  latent_codes = []
  image_names = []
  print('Building inverter')
  # inverter = build_inverter(model_name=model_name)

  for image_name in natsorted(os.listdir(source_dir)):
    if image_name.split('.')[-1].lower() is 'jpg' or 'png' or 'jpeg' :
      
      if not need_align:
        mani_image, mani_image_name = align(inverter, os.path.join(source_dir,image_name))
        if mani_image is not None:
          if mani_image.shape[2] == 4:
            mani_image = mani_image[:, :, :3]
      else:
         mani_image = plt.imread(os.path.join(source_dir,image_name))

      latent_code, _ , ssim_loss= invert(inverter, mani_image)
      if ssim_loss>threshold:
        image_names.append(image_name)
        latent_codes.append(latent_code)
  return latent_codes,image_names






def load_image(path):
  """Loads an image from disk.

  NOTE: This function will always return an image with `RGB` channel order for
  color image and pixel range [0, 255].

  Args:
    path: Path to load the image from.

  Returns:
    An image with dtype `np.ndarray` or `None` if input `path` does not exist.
  """
  if not os.path.isfile(path):
    return None

  image = Image.open(path)
  return image

def flatten(t):
    return [item for sublist in t for item in sublist]

    
def load_images_from_dir(dspth,align_size = 256, need_align = False , use_hog = False):

  images = []
  image_names =  natsorted(os.listdir(dspth))
  aligned_images_names = []


  for image_name in natsorted(os.listdir(dspth)):
    if image_name.split('.')[-1].lower() is 'jpg' or 'png' or 'jpeg' :
      if need_align:
        aligned_image , aligned_name  = align_face((os.path.join(dspth,image_name)),align_size=align_size , use_hog = use_hog )

      else:
        aligned_image = plt.imread(os.path.join(dspth,image_name))
        aligned_image = cv2.resize(aligned_image , (align_size,align_size))
        aligned_name = image_name


      images.append(aligned_image)
      if type(aligned_image) == list :
        images = flatten(images)

      aligned_images_names.append(aligned_name)
      if type(aligned_image) == list:
        aligned_images_names = flatten(aligned_images_names)

      

  return images,aligned_images_names


def imshow(images, col, viz_size=256):
  """Shows images in one figure."""
  num, height, width, channels = images.shape
  assert num % col == 0
  row = num // col

  fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

  for idx, image in enumerate(images):
    i, j = divmod(idx, col)
    y = i * viz_size
    x = j * viz_size
    if height != viz_size or width != viz_size:
      image = cv2.resize(image, (viz_size, viz_size))
    fused_image[y:y + viz_size, x:x + viz_size] = image

  fused_image = np.asarray(fused_image, dtype=np.uint8)
  data = io.BytesIO()
  if channels == 4:
    Image.fromarray(fused_image).save(data, 'png')
  elif channels == 3:
    Image.fromarray(fused_image).save(data, 'jpeg')
  else:
    raise ValueError('Image channel error')
  im_data = data.getvalue()
  disp = IPython.display.display(IPython.display.Image(im_data))
  return disp


def get_landmarks(images):
  landmarks = []
  fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
  for img in images:
    pred = fa.get_landmarks(img)
    pred = np.array(pred)
    pred.resize((68,2))
    landmarks.append(pred)
    
  return landmarks


import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import cv2
import matplotlib.pyplot as plt
import pickle
from pylab import *

from models.hypergan_models.model import Model as hyperganModel
import utils.segment
import utils.inpainting
from utils.logger import setup_logger
from utils import util


def parse_args():
  """Parses arguments."""
  
  parser = argparse.ArgumentParser()

  parser.add_argument('--test_dir', type=str, default = './test_data',
                      help='directory of images to invert.')

  parser.add_argument('-o', '--output_dir', type=str, default='./results',
                      help='Directory to save the results. If not specified, '
                           '`./results/'
                           'will be used by default.')

  parser.add_argument('--pretrained_dir', type=str, default = './pretrained_models',
                      help='Directory tof pretraied models. If not specified, '
                           '`./pretrained_models/'
                           'will be used by default.')
  
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')

  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')

  parser.add_argument('--loss_weight_perceptual', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')

  parser.add_argument('--loss_weight_ssim', type=float, default=3.0,
                      help='The perceptual loss scale for optimization. '
                           '(default: 3.0)')

  parser.add_argument('--loss_weight_regularization', type=float, default=0.2,
                      help='The regularization loss weight for optimization. '
                           '(default: 0.2)')
  
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')

  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  
  return parser.parse_args()



def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    assert os.path.exists(args.test_dir)
    model_name = 'styleganinv_ffhq256'  
    MODEL_DIR = os.path.join(args.pretrained_dir)
    os.makedirs(MODEL_DIR, exist_ok=True)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inpainting_save_path = os.path.join(output_dir,'inpainting/')

    latent_codes_save_path = os.path.join(output_dir,'inversion/')
    if not os.path.exists(latent_codes_save_path):
         os.makedirs(latent_codes_save_path)
    
     

    logger,logfile_name = setup_logger(output_dir, 'hairGAN.log', 'logger')
    logger.info(f"logging into {logfile_name}")

    images,_ = util.load_images_from_dir(args.test_dir)
    logger.info(f"found {len(images)} images in the folder")

    logger.info('Loading models for stylegan inversion.')

    logger.info('Loading inverter model.')
    inverter = util.build_inverter(model_name = model_name,logger =logger,iteration = 200,pretrained_weights = os.path.join(MODEL_DIR,"vgg16.pth"))

    logger.info('Loading generator model.')
    latent_code_to_image_generator = util.get_generator(model_name = model_name)

    logger.info('Loading models for hypergan inpainting.')

    hypergan_model = hyperganModel()
    hypergan_generator = hypergan_model.build_generator()
    checkpoint = tf.train.Checkpoint(generator=hypergan_generator)
    checkpoint.restore(os.path.join(args.pretrained_dir, "ckpt-25"))
    inpainting_results = utils.inpainting.inpainting(args.test_dir,generator=hypergan_generator,savepath=inpainting_save_path) 
    logger.info(f"succesfully completed inpainting on {len(inpainting_results)} / {len(images)} images")



    logger.info("starting inversion!")
    latent_codes,inverted_image_names   = util.batch_invert(inverter=inverter,source_dir=inpainting_save_path)
    latent_codes = np.squeeze(np.array(latent_codes),1)
    np.save(os.path.join(latent_codes_save_path,"latent_codes.npy"),latent_codes)
    generated_images = latent_code_to_image_generator.easy_synthesize(latent_codes, **{'latent_space_type': 'wp'})['image']

    new_images = []


    for i in range(0,generated_images.shape[0]):
         new_images.append(generated_images[i])
         plt.imsave(os.path.join(latent_codes_save_path,inverted_image_names[i]),generated_images[i])

     
     


if __name__ == '__main__':
  main()













    



  
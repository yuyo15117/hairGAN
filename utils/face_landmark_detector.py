
import os
import bz2
import requests
import dlib
import numpy as np
from PIL import Image
import scipy.ndimage


class FaceLandmarkDetector(object):
  """Class of face landmark detector."""

  def __init__(self, align_size=256, use_hog = False, enable_padding=True, model_name = 'styleganinv_ffhq256' ,model_dir = 'pretrained_models' ,landmark_model_name = 'shape_predictor_68_face_landmarks.dat', mmod_model_name='mmod_human_face_detector.dat'):
    """Initializes face detector and landmark detector.

  Args:
    align_size: Size of the aligned face if performing face alignment.
    (default: 1024)
    enable_padding: Whether to enable padding for face alignment (default:
    True)
  """
    # Download models if needed.
    self.use_hog  = use_hog

    self.model_name = model_name
    self.model_dir = os.path.join(model_dir)
    os.makedirs(self.model_dir, exist_ok=True)






    self.landmark_model_path = os.path.join(self.model_dir, landmark_model_name)

    self.landmark_model_url = f'http://dlib.net/files/{landmark_model_name}.bz2'

    if not os.path.exists(self.landmark_model_path):
      data = requests.get(self.landmark_model_url)
      data_decompressed = bz2.decompress(data.content)
      with open(self.landmark_model_path, 'wb') as f:
        f.write(data_decompressed)

    if use_hog:
      self.face_detector = dlib.get_frontal_face_detector()
    else:
      self.mmod_face_detector_model_path = os.path.join(self.model_dir, mmod_model_name)
      self.face_detector = dlib.cnn_face_detection_model_v1(self.mmod_face_detector_model_path)


    self.landmark_detector = dlib.shape_predictor(self.landmark_model_path)
    self.align_size = align_size
    self.enable_padding = enable_padding

  def detect(self,image_path = None):
    """Detects landmarks from the given image.

  This function will first perform face detection on the input image. All
  detected results will be grouped into a list. If no face is detected, an
  empty list will be returned.

  For each element in the list, it is a dictionary consisting of `image_path`,
  `bbox` and `landmarks`. `image_path` is the path to the input image. `bbox`
  is the 4-element bounding box with order (left, top, right, bottom), and
  `landmarks` is a list of 68 (x, y) points.

  Args:
    image_path: Path to the image to detect landmarks from.

  Returns:
    A list of dictionaries, each of which is the detection results of a
    particular face.
  """
    results = []

    # image_ = np.array(image)
    images = dlib.load_rgb_image(image_path)

    # Face detection (1 means to upsample the image for 1 time.)
    bboxes = self.face_detector(images, 2)
    # Landmark detection
    for bbox in bboxes:
      if not self.use_hog:
        bbox = bbox.rect
      landmarks = []
      for point in self.landmark_detector(images, bbox).parts():
        landmarks.append((point.x, point.y))
      results.append({
          'image_path': image_path,
          'bbox': (bbox.left(), bbox.top(), bbox.right(), bbox.bottom()),
          'landmarks': landmarks,
      })
    return results

  def align(self, face_info):
    """Aligns face based on landmark detection.

  The face alignment process is borrowed from
  https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py,
  which only supports aligning faces to square size.

  Args:
    face_info: Face information, which is the element of the list returned by
    `self.detect()`.

  Returns:
    A `np.ndarray`, containing the aligned result. It is with `RGB` channel
    order.
  """
    img = Image.open(face_info['image_path'])

    landmarks = np.array(face_info['landmarks'])
    eye_left = np.mean(landmarks[36: 42], axis=0)
    eye_right = np.mean(landmarks[42: 48], axis=0)
    eye_middle = (eye_left + eye_right) / 2
    eye_to_eye = eye_right - eye_left
    mouth_middle = (landmarks[48] + landmarks[54]) / 2
    eye_to_mouth = mouth_middle - eye_middle

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_middle + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / self.align_size * 0.5))
    if shrink > 1:
      rsize = (int(np.rint(float(img.size[0]) / shrink)),
               int(np.rint(float(img.size[1]) / shrink)))
      img = img.resize(rsize, Image.ANTIALIAS)
      quad /= shrink
      qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
      img = img.crop(crop)
      quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0),
           max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if self.enable_padding and max(pad) > border - 4:
      pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
      img = np.pad(np.float32(img),
                   ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                   'reflect')
      h, w, _ = img.shape
      y, x, _ = np.ogrid[:h, :w, :1]
      mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                         np.float32(w - 1 - x) / pad[2]),
                        1.0 - np.minimum(np.float32(y) / pad[1],
                                         np.float32(h - 1 - y) / pad[3]))
      blur = qsize * 0.02
      blurred_image = scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img
      img += blurred_image * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
      img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
      img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
      quad += pad[:2]

    # Transform.
    img = img.transform((self.align_size * 4, self.align_size * 4), Image.QUAD,
                        (quad + 0.5).flatten(), Image.BILINEAR)
    img = img.resize((self.align_size, self.align_size), Image.ANTIALIAS)

    return np.array(img)


# def align_face(image_path, align_size=256):
#   """Aligns a given face."""
#   model = FaceLandmarkDetector(align_size)
#   face_infos = model.detect(image_path)
#   if len(face_infos) != 0:
#     face_infos = face_infos[0]
#     img = model.align(face_infos)

#   else:
#     return None
#   return img


# def build_inverter(model_name, iteration=100, regularization_loss_weight=2 , ):
#   """Builds inverter"""
#   inverter = StyleGANInverter(
#       model_name,
#       learning_rate=0.01,
#       iteration=iteration,
#       reconstruction_loss_weight=1.0,
#       perceptual_loss_weight=5e-5,
#       regularization_loss_weight=regularization_loss_weight)
#   return inverter

# def build_inverter2(model_name, iteration=100, regularization_loss_weight=0.2 , loss_weight_ssim=3 ):
#   """Builds inverter"""
#   inverter2 = StyleGANInverter(
#       model_name,
#       learning_rate=0.01,
#       iteration=iteration,
#       reconstruction_loss_weight=1.0,
#       perceptual_loss_weight=5e-5,
#       regularization_loss_weight=regularization_loss_weight,
#       loss_weight_ssim = loss_weight_ssim
      
#       )
#   return inverter2



# def get_generator(model_name):
#   """Gets model by name"""
#   return build_generator(model_name)


# def align(inverter, image_path):
#   """Aligns an unloaded image."""
#   aligned_image = align_face(image_path,
#                              align_size=inverter.G.resolution)
#   return aligned_image


# def invert(inverter, image):
#   """Inverts an image."""
#   latent_code, reconstruction , ssimLoss = inverter.easy_invert(image, num_viz=1)
#   return latent_code, reconstruction, ssimLoss


# def diffuse(inverter, target, context, left, top, width, height):
#   """Diffuses a target image to a context image."""
#   center_x = left + width // 2
#   center_y = top + height // 2
#   _, diffusion = inverter.easy_diffuse(target=target,
#                                        context=context,
#                                        center_x=center_x,
#                                        center_y=center_y,
#                                        crop_x=width,
#                                        crop_y=height,
#                                        num_viz=1)
#   return diffusion


# def load_image(path):
#   """Loads an image from disk.

#   NOTE: This function will always return an image with `RGB` channel order for
#   color image and pixel range [0, 255].

#   Args:
#     path: Path to load the image from.

#   Returns:
#     An image with dtype `np.ndarray` or `None` if input `path` does not exist.
#   """
#   if not os.path.isfile(path):
#     return None

#   image = Image.open(path)
#   return image

# def imshow(images, col, viz_size=256):
#   """Shows images in one figure."""
#   num, height, width, channels = images.shape
#   assert num % col == 0
#   row = num // col

#   fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

#   for idx, image in enumerate(images):
#     i, j = divmod(idx, col)
#     y = i * viz_size
#     x = j * viz_size
#     if height != viz_size or width != viz_size:
#       image = cv2.resize(image, (viz_size, viz_size))
#     fused_image[y:y + viz_size, x:x + viz_size] = image

#   fused_image = np.asarray(fused_image, dtype=np.uint8)
#   data = io.BytesIO()
#   if channels == 4:
#     Image.fromarray(fused_image).save(data, 'png')
#   elif channels == 3:
#     Image.fromarray(fused_image).save(data, 'jpeg')
#   else:
#     raise ValueError('Image channel error')
#   im_data = data.getvalue()
#   disp = IPython.display.display(IPython.display.Image(im_data))
#   return disp

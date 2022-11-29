import random
import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch
import MinkowskiEngine as ME
from scipy.linalg import expm, norm

# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Feature transformations
##############################
class RGBShift(object):
  """Add random color to the image, input must be an array in [0,255]"""

  def __init__(self, shift_limit=(-20, 20)):
    
    self.shift_limit = shift_limit

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      shift = np.random.uniform(*self.shift_limit, size=3)
      feats[:, :3] = np.clip(shift + feats[:, :3], 0, 255)
    return coords, feats, labels


class RandomBrightnessContrast(object):
      
  def __init__(self, brightness_limit=(-0.2, 0.2), contrast_limit=(0.8, 1.2)):
    self.brightness_limit = brightness_limit
    self.contrast_limit = contrast_limit

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      brightness = np.random.uniform(*self.brightness_limit)
      contrast = np.random.uniform(*self.contrast_limit)
      feats[:, :3] = np.clip(feats[:, :3]*contrast + 255*brightness, 0, 255)
    return coords, feats, labels


class ChromaticJitter(object):

  def __init__(self, std=0.05):
    self.std = std

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      noise = np.random.randn(feats.shape[0], 3)
      noise *= self.std * 255
      feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
    return coords, feats, labels


##############################
# Coordinate transformations
##############################
class RandomTranslateRotateScale(object):
  
  def __init__(self,
               translation_aug_prob=1.0,
               rotation_aug_bound=((-np.pi/24, np.pi/24), (-np.pi/24, np.pi/24), (-np.pi, np.pi)),
               scale_aug_bound=(0.9, 1.1),
               is_temporal=False):
    
    self.translation_aug_prob = translation_aug_prob
    self.rotation_aug_bound = rotation_aug_bound
    self.scale_aug_bound = scale_aug_bound
    self.is_temporal = is_temporal
    
  def __call__(self, coords, feats, labels):
    # Translate
    if self.is_temporal:
      stacked_coords = np.vstack(coords)
      mean, min, max = stacked_coords.mean(0), stacked_coords.min(0), stacked_coords.max(0)
    else:
      mean, min, max = coords.mean(0), coords.min(0), coords.max(0)
    
    translate = -mean
    if random.random() < self.translation_aug_prob:
      translate += np.random.uniform(min-mean, max-mean)/2
        
    # Scale
    scale_matrix = np.eye(3)
    scale = np.random.uniform(*self.scale_aug_bound)
    np.fill_diagonal(scale_matrix, scale)
    
    # Rotate
    rot_mats = []
    for axis_ind, rot_bound in enumerate(self.rotation_aug_bound):
      axis = np.zeros(3)
      axis[axis_ind] = 1
      theta = np.random.uniform(*rot_bound)
      single_rotation = expm(np.cross(np.eye(3), axis / norm(axis) * theta))
      rot_mats.append(single_rotation)
    # Use random order
    rotation_matrix = rot_mats[2] @ rot_mats[1] @ rot_mats[0]
    
    if self.is_temporal:
      coords = [(C+translate) @ scale_matrix @ rotation_matrix for C in coords]
    else:
      coords = (coords+translate) @ scale_matrix @ rotation_matrix
    
    return coords, feats, labels


class RandomDropout(object):

  def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.2):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.dropout_ratio = dropout_ratio
    self.dropout_application_ratio = dropout_application_ratio

  def __call__(self, coords, feats, labels):
    if random.random() < self.dropout_application_ratio:
      N = len(coords)
      inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
      return coords[inds], feats[inds], labels[inds]
    return coords, feats, labels


class RandomHorizontalFlip(object):

  def __init__(self, upright_axis, is_temporal):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      for curr_ax in self.horz_axes:
        if random.random() < 0.5:
          coord_max = np.max(coords[:, curr_ax])
          coords[:, curr_ax] = coord_max - coords[:, curr_ax]
    return coords, feats, labels


class ElasticDistortion:

  def __init__(self, distortion_params=((0.2, 0.4), (0.8, 1.6))):
    self.distortion_params = distortion_params

  def elastic_distortion(self, coords, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

      pointcloud: numpy array of (number of points, at least 3 spatial dims)
      granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
      magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
      noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                   (noise_dim - 2), noise_dim)
    ]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    coords += interp(coords) * magnitude
    return coords

  def __call__(self, coords, feats, labels):
    if self.distortion_params is not None:
      if random.random() < 0.95:
        for granularity, magnitude in self.distortion_params:
          coords = self.elastic_distortion(coords, granularity, magnitude)
    return coords, feats, labels


class Compose(object):
  """Composes several transforms together."""

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, *args):
    for t in self.transforms:
      if type(args) is tuple:
        args = t(*args)
      else:
        args = t(args)
    return args


class cfl_collate_fn_factory:
  """Generates collate function for coords, feats, labels.

    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                       size so that the number of input coordinates is below limit_numpoints.
  """

  def __init__(self, limit_numpoints, return_inverse):
    self.limit_numpoints = limit_numpoints
    self.return_inverse = return_inverse

  def __call__(self, list_data):
    coords_b, feats_b, labels_b, original_labels_b, inverse_maps_b = [], [], [], [], []
    
    if self.return_inverse:
      coords, feats, labels, original_labels, inverse_maps = list(zip(*list_data))
    else:
      coords, feats, labels = list(zip(*list_data))
    
    batch_num_points = 0
    for batch_id, _ in enumerate(coords):
      num_points = coords[batch_id].shape[0]
      
      if self.limit_numpoints and (batch_num_points+num_points) > self.limit_numpoints:
        logging.warning("Skipping a scene")
        continue
          
      coords_b.append(torch.from_numpy(coords[batch_id]).int())
      feats_b.append(torch.from_numpy(feats[batch_id]).float())
      labels_b.append(torch.from_numpy(labels[batch_id]).int())
      if self.return_inverse:
        original_labels_b.append(original_labels[batch_id])
        inverse_maps_b.append(inverse_maps[batch_id])
      batch_num_points += num_points

    # Concatenate all lists
    coords_b, feats_b, labels_b = ME.utils.sparse_collate(coords_b, feats_b, labels_b)
    return_args = [coords_b, feats_b, labels_b]
    if self.return_inverse:
      return_args.append(original_labels_b)
      return_args.append(inverse_maps_b)
    
    return tuple(return_args)
  
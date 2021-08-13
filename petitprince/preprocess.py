import numpy as np
import torch
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.input_data import NiftiMasker
from nilearn.masking import intersect_masks
from scipy.stats import zscore

def preprocess(data, gmask, normalize=True, range01=False, mask=None):
  """Preprocess fMRI timeseries.
  Inputs:
      data: 4D fMRI input (3D image x time)
      gmask: gray matter mask
      normalize: scale using overall mean and std
      range01: ensure the data are within -1 to 1 range
      mask: the mask to be applied. Possible values:
        - vis: visual cortex
        - aud: auditory cortex
        - ifg: inferior frontal gyrus
        - random: take random 1000 voxels
        The masks are currently bilateral. The gray matter mask is always applied.
  Output: preprocessed data of size n x t,
     where n = number of voxels, t = number of timepoints
  """

  if mask:
    if mask=='random':
      vox_is = np.random.randint(0, high=data1.shape[1], size=1000)
      data = data[:, vox_is]
      mask_image = gmask
    else:
      mask_image = math_img('img > 0.99', img=eval('mask_'+mask))
      gmask = resample_to_img(gmask, mask_image, interpolation='nearest')
      mask_image = intersect_masks([mask_image, gmask])
  else:
    mask_image=gmask
  masker = NiftiMasker(mask_img=mask_image).fit()
  data = masker.transform(data)

  if normalize:
    data = torch.Tensor(zscore(data, axis=0))
  if range01:
    data = torch.Tensor(data)
    data = F.normalize(data, dim=0)

  # filter out voxels with nans
  data = data[:,~torch.any(data.isnan(),dim=0)]

  return data.T

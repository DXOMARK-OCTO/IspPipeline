import rawpy
import imageio
import math
import numpy as np
import scipy.ndimage

# Function that removes Black Level And Normalize between 0 and 1
def ispBlackLevel(img_raw, bl, wl) -> np.ndarray :
  img_out = img_raw.copy()
  img_out = (img_out - bl) / (wl - bl)
  img_out[img_out < 0] = 0
  return img_out

# Function that apply a gain (exposure compensation)
def ispGain(img_in, gain) -> np.ndarray :
  return gain*img_in.copy().clip(0, 1)

def ispSplitColors(img_in) -> np.ndarray:
  h, w = img_in.shape
  img_out = np.zeros((h, w, 3))
  # Red
  img_out[0::2, 0::2, 0] = img_in[0::2, 0::2]
  # Green
  img_out[1::2, 0::2, 1] = img_in[1::2, 0::2]
  img_out[0::2, 1::2, 1] = img_in[0::2, 1::2]
  # Blue
  img_out[1::2, 1::2, 2] = img_in[1::2, 1::2]
  return img_out

def ispAdvancedDemosaicing(img_in) -> np.ndarray:
  img_out = np.empty_like(img_in)
  h, w, _ = img_in.shape
  # Compute two proposals for G
  kernel_dir = np.array([0.5, 1.0, 0.5])
  green_H = scipy.ndimage.correlate1d(img_in[:, :, 1], kernel_dir, axis=1, mode='mirror')
  green_V = scipy.ndimage.correlate1d(img_in[:, :, 1], kernel_dir, axis=0, mode='mirror')
  # Compute variation and select best proposal for G
  kernel_var  = np.array([-1, 0, 1])
  var_H = np.abs(scipy.ndimage.correlate1d(green_H, kernel_var, axis=1, mode='mirror'))
  var_V = np.abs(scipy.ndimage.correlate1d(green_V, kernel_var, axis=0, mode='mirror'))
  img_out[:, :, 1] = (var_H > var_V).choose(green_H, green_V)
  # Compute deltaR and deltaB
  deltaR = np.zeros((h, w))
  deltaR[0::2, 0::2] = img_in[0::2, 0::2, 0] - img_out[0::2, 0::2, 1]
  deltaB = np.zeros((h, w))
  deltaB[1::2, 1::2] = img_in[1::2, 1::2, 2] - img_out[1::2, 1::2, 1]
  # Linear interpolation for deltaR and deltaB
  kernel_RB = np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])
  deltaR = scipy.ndimage.correlate(deltaR, kernel_RB, mode='mirror')
  deltaB = scipy.ndimage.correlate(deltaB, kernel_RB, mode='mirror')
  # Compute R and B
  img_out[:, :, 0] = (img_out[:, :, 1] + deltaR).clip(0, 1)
  img_out[:, :, 2] = (img_out[:, :, 1] + deltaB).clip(0, 1)
  return img_out

# Applies some White Balance Gains on an RGB image
def ispWhiteBalance(img_in,wbg) -> np.ndarray :
  return img_in * wbg

# Applies a 3x3 Matrix on an RGB image
def ispColorMatrix(img_in, color_matrix) -> np.ndarray:
  return np.dot(img_in, color_matrix.T).clip(0, 1)

# Applies a power curve on an RGB image
def ispApplyGamma(img_in, gamma) -> np.ndarray:
  return img_in ** (gamma)

def makeSatMatrix(sat_param) -> np.ndarray:
  gray_matrix = np.array([
    [0.299, 0.587, 0.114],
    [0.299, 0.587, 0.114],
    [0.299, 0.587, 0.114]
  ])
  return gray_matrix + sat_param * (np.eye(3) - gray_matrix)

def makeUnsharpMaskFilter(strengthParameter) -> np.ndarray:
    gaussianFilter = np.array([1, 4, 6, 4, 1])/16
    gaussianFilterKernel = np.outer(gaussianFilter,gaussianFilter)
    identityFilter = np.array([0, 0, 1, 0, 0])
    identityFilterKernel = np.outer(identityFilter,identityFilter)
    
    unsharpMaskKernel = gaussianFilterKernel + strengthParameter * ( identityFilterKernel - gaussianFilterKernel )

    print(unsharpMaskKernel)

    return unsharpMaskKernel

def ispApplyKernel(img_in, kernel) -> np.ndarray:
  kernel_RB = np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])
  img_out = np.empty_like(img_in)
  img_out[:, :, 0] = scipy.ndimage.correlate(img_in[:, :, 0], kernel, mode='nearest').clip(0, 1)
  img_out[:, :, 1] = scipy.ndimage.correlate(img_in[:, :, 1], kernel, mode='nearest').clip(0, 1)
  img_out[:, :, 2] = scipy.ndimage.correlate(img_in[:, :, 2], kernel, mode='nearest').clip(0, 1)
  return img_out

def isp(img_in, blackLevel, gain, WBGains, colorMatrix, whiteLevel) -> np.ndarray :

  img_bl = ispBlackLevel(img_in, blackLevel, whiteLevel)
  img_gain = ispGain(img_bl, gain)
  img_split = ispSplitColors(img_gain)
  img_wb = ispWhiteBalance(img_split, WBGains)
  img_dms = ispAdvancedDemosaicing(img_wb)
  img_ccm = ispColorMatrix(img_dms,colorMatrix)
  img_gamma = ispApplyGamma(img_ccm, 1/2.2)

  satMatrix = makeSatMatrix(1.5)
  
  img_ccm = ispColorMatrix(img_gamma,satMatrix)

  usmKernel = makeUnsharpMaskFilter(2)

  img_out = ispApplyKernel(img_ccm,usmKernel)

  return img_out

if __name__ == '__main__':
    raw = rawpy.imread(r".\SonyA7S3\ISO100.dng")

    bp = raw.raw_pattern
    bl = raw.black_level_per_channel
    wl  = raw.white_level
    wb = np.array(raw.camera_whitebalance)
    cm = raw.color_matrix[0:3,0:3]
    gain = 4

    # cropRaw = np.array(raw.raw_image.copy()).reshape((raw.sizes.raw_height, raw.sizes.raw_width)).astype('float')[900:1300,1950:2350]
    cropRaw = np.array(raw.raw_image.copy()).reshape((raw.sizes.raw_height, raw.sizes.raw_width)).astype('float')

    cropIsp = isp(cropRaw, bl[0], 2, wb[:3], cm, wl)

    outimg = cropIsp.copy()
    outimg[outimg < 0] = 0
    outimg[outimg > 1] = 1
    outimg = outimg * 255
    imageio.imwrite("ISO100_ISP.jpg", outimg.astype('uint8'))
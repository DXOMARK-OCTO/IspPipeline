import rawpy
import imageio
import math
import numpy as np

# Function that removes Black Level
def ispBL(img_in, blc, bayer_pattern) -> np.ndarray :
  img_out = img_in.copy()
  h, w = img_out.shape
  for y in range(0, h, 2):
    for x in range(0, w, 2):
      img_out[y + 0, x + 0] -= blc[bayer_pattern[0, 0]]
      img_out[y + 0, x + 1] -= blc[bayer_pattern[0, 1]]
      img_out[y + 1, x + 0] -= blc[bayer_pattern[1, 0]]
      img_out[y + 1, x + 1] -= blc[bayer_pattern[1, 1]]
  img_out[img_out < 0] = 0
  return img_out

# Function that apply a gain (exposure compensation)
def ispEC(img_in, gain) -> np.ndarray :
	img_out = gain*img_in.copy()
	return img_out

def ispBilinearDM(img_in) -> np.ndarray :
# this function only works for RGGB patterns
  h, w = img_in.shape
  img_out = np.zeros((h, w, 3))
  for y in range(2, h - 2, 2):
    for x in range(2, w - 2, 2):
      R, G, B = 0, 1, 2

      ystart = y - 2
      yend = y + 4
      xstart = x - 2
      xend = x + 4
      red     = img_in[ ( ystart + 0 ) : ( yend + 0 ) : 2 , ( xstart + 0 ) : ( xend + 0) : 2 ]    # matrix of 3x3 red pixels
      greenTR = img_in[ ( ystart + 0 ) : ( yend + 0 ) : 2 , ( xstart + 1 ) : ( xend + 1) : 2 ]    # matrix of 3x3 green Top Right pixels
      greenBL = img_in[ ( ystart + 1 ) : ( yend + 1 ) : 2 , ( xstart + 0 ) : ( xend + 0) : 2 ]    # matrix of 3x3 green Bottom Left pixels
      blue    = img_in[ ( ystart + 1 ) : ( yend + 1 ) : 2 , ( xstart + 1 ) : ( xend + 1) : 2 ]    # matrix of 3x3 blue Top Right pixels
        
    # Fill The Code

      img_out[y    , x     , R] = red[1][1]
      img_out[y    , x     , G] = ( greenTR[1][1] + greenTR[0][1] + greenBL[1][1] + greenBL[1][0] ) / 4
      img_out[y    , x     , B] = ( blue[0][0] + blue[0][1] + blue[1][1] + blue[1][0] ) / 4
 
      img_out[y + 1, x     , R] = ( red[1][1] + red[2][1] ) / 2
      img_out[y + 1, x     , G] = greenBL[1][1]
      img_out[y + 1, x     , B] = ( blue[1][0] + blue[1][1] ) / 2

      img_out[y + 1, x + 1 , R] = ( red[1][1] + red[1][2] + red[2][1] + red[2][2] ) / 4
      img_out[y + 1, x + 1 , G] = ( greenTR[1][1] + greenTR[2][1] + greenBL[1][1] + greenBL[1][2] ) / 4
      img_out[y + 1, x + 1 , B] = blue[1][1]

      img_out[y    , x + 1 , R] = ( red[1][1] + red[1][2] ) / 2
      img_out[y    , x + 1 , G] = greenTR[1][1]
      img_out[y    , x + 1 , B] = ( blue[0][1] + blue[1][1] ) / 2

    # *************

  return img_out

# Applies some White Balance Gains on an RGB image
def ispWB(img_in,wbg) -> np.ndarray :
  h, w, c = img_in.shape
  img_out = img_in.copy().flatten().reshape((-1, 3))
  for index, pixel in enumerate(img_out):
    pixel = pixel * wbg
    img_out[index] = pixel
  img_out = img_out.reshape((h,w,3))
  return img_out

# Applies some White Balance Gains on an raw image
def ispRawWB(img_in,wbg) -> np.ndarray :
# this function only works for RGGB patterns
  h, w = img_in.shape
  img_out = np.zeros((h, w))
  for y in range(0, h, 2):
    for x in range(0, w, 2):
      colors = [0, 0, 0, 0]
      img_out[y + 0, x + 0] = wbg[0]*img_in[y + 0, x + 0]  # Red
      img_out[y + 0, x + 1] = wbg[1]*img_in[y + 0, x + 1]  # Green Top Right
      img_out[y + 1, x + 0] = wbg[1]*img_in[y + 1, x + 0]  # Green Bottom Left
      img_out[y + 1, x + 1] = wbg[2]*img_in[y + 1, x + 1]  # Blue

  return img_out

# Bilinear Green upsampling.
def bilinearGreenUpsamplingDM(img_in) -> np.ndarray :
# this function only works for RGGB patterns
  img_out = np.zeros((h, w,))
  for y in range(2, h - 2, 2):
    for x in range(2, w - 2, 2):

      ystart = y - 2
      yend = y + 4
      xstart = x - 2
      xend = x + 4

      greenTR = img_in[ ( ystart + 0 ) : ( yend + 0 ) : 2 , ( xstart + 1 ) : ( xend + 1) : 2 ]    # matrix of 3x3 green Top Right pixels
      greenBL = img_in[ ( ystart + 1 ) : ( yend + 1 ) : 2 , ( xstart + 0 ) : ( xend + 0) : 2 ]    # matrix of 3x3 green Bottom Left pixels
        
    # Fill The Code
      # Interpolating the green (~ Luminance) channel by bilinear interpolation

      img_out[y    , x    ] = ( greenTR[1][1] + greenTR[0][1] + greenBL[1][1] + greenBL[1][0] ) / 4
      img_out[y + 1, x    ] = greenBL[1][1]
      img_out[y + 1, x + 1] = ( greenTR[1][1] + greenTR[2][1] + greenBL[1][1] + greenBL[1][2] ) / 4
      img_out[y    , x + 1] = greenTR[1][1]
  
  return img_out

# Bilinear Green upsampling.
def hamiltonAdamsGreenUpsamplingDM(img_in) -> np.ndarray :
# this function only works for RGGB patterns
  h, w = img_in.shape
  img_out = np.zeros((h, w,))
  for y in range(2, h - 2, 2):
    for x in range(2, w - 2, 2):

      ystart = y - 2
      yend = y + 4
      xstart = x - 2
      xend = x + 4

      greenTR = img_in[ ( ystart + 0 ) : ( yend + 0 ) : 2 , ( xstart + 1 ) : ( xend + 1) : 2 ]    # matrix of 3x3 green Top Right pixels
      greenBL = img_in[ ( ystart + 1 ) : ( yend + 1 ) : 2 , ( xstart + 0 ) : ( xend + 0) : 2 ]    # matrix of 3x3 green Bottom Left pixels
      red     = img_in[ ( ystart + 0 ) : ( yend + 0 ) : 2 , ( xstart + 0 ) : ( xend + 0) : 2 ]    # matrix of 3x3 red pixels
      blue    = img_in[ ( ystart + 1 ) : ( yend + 1 ) : 2 , ( xstart + 1 ) : ( xend + 1) : 2 ]    # matrix of 3x3 blue Top Right pixels

    # Fill The Code
      # Greens that do not need interpolation
      img_out[y + 1, x    ] = greenBL[1][1]
      img_out[y    , x + 1] = greenTR[1][1]
      
      #Calcultate Gradients for Green interpolation at Red pixel
      redHorDiff = red[1][1] - red[1][0] + red[1][1] - red[1][2]
      horGrad = abs(greenTR[1][0] - greenTR[1][1]) + abs( redHorDiff )
      redVerDiff = red[1][1] - red[0][1] + red[1][1] - red[2][1]
      verGrad = abs(greenBL[0][1] - greenBL[1][1]) + abs( redVerDiff )

      # Directionnal Interpolation at Red pixel
      if horGrad > verGrad:
        img_out[y    , x    ] = ( greenBL[1][1] + greenBL[0][1] ) / 2 + ( redVerDiff ) / 4
      elif verGrad > horGrad:
        img_out[y    , x    ] = ( greenTR[1][1] + greenTR[1][0] ) / 2 + ( redHorDiff ) / 4
      else:
        img_out[y    , x    ] = ( greenTR[1][1] + greenTR[0][1] + greenBL[1][1] + greenBL[1][0] ) / 4 + ( redVerDiff + redHorDiff ) / 8

      #Calcultate Gradients for Green interpolation at Blue pixel
      blueHorDiff = blue[1][1] - blue[1][0] + blue[1][1] - blue[1][2]
      horGrad = abs(greenBL[1][1] - greenBL[1][2]) + abs( blueHorDiff )
      blueVerDiff = blue[1][1] - blue[0][1] + blue[1][1] - blue[2][1]
      verGrad = abs(greenTR[1][1] - greenTR[2][1]) + abs( blueVerDiff )

      # Directionnal Interpolation at Blue pixel
      if horGrad > verGrad:
        img_out[y + 1, x + 1] = ( greenTR[1][1] + greenTR[2][1] ) / 2 + ( blueVerDiff ) / 4
      elif verGrad > horGrad:
        img_out[y + 1, x + 1] = ( greenBL[1][1] + greenBL[1][2] ) / 2 + ( blueHorDiff ) / 4
      else:
        img_out[y + 1, x + 1] = ( greenTR[1][1] + greenTR[2][1] + greenBL[1][1] + greenBL[1][2] ) / 4 + ( blueVerDiff + blueHorDiff ) / 8

  img_out[img_out < 0] = 0

  return img_out

# Bilinear Chroma upsampling.
def bilinearChromaGreenGuidedUpsamplingDM(img_in, img_green) -> np.ndarray :
# this function only works for RGGB patterns
  h, w = img_in.shape
  img_out = np.zeros((h, w, 3))
  for y in range(2, h - 2, 2):
    for x in range(2, w - 2, 2):

      R, G, B = 0, 1, 2

      ystart = y - 2
      yend = y + 4
      xstart = x - 2
      xend = x + 4

      # matrix of 3x3 Chroma Red pixels
      chromaRed     = img_in[ ( ystart + 0 ) : ( yend + 0 ) : 2 , ( xstart + 0 ) : ( xend + 0) : 2 ]  -  img_green[ ( ystart + 0 ) : ( yend + 0 ) : 2 , ( xstart + 0 ) : ( xend + 0) : 2 ]
        # matrix of 3x3 Chroma Blue pixels
      chromaBlue    = img_in[ ( ystart + 1 ) : ( yend + 1 ) : 2 , ( xstart + 1 ) : ( xend + 1) : 2 ]  -  img_green[ ( ystart + 1 ) : ( yend + 1 ) : 2 , ( xstart + 1 ) : ( xend + 1) : 2 ]

      red     = img_in[ ( ystart + 0 ) : ( yend + 0 ) : 2 , ( xstart + 0 ) : ( xend + 0) : 2 ]    # matrix of 3x3 red pixels
      blue    = img_in[ ( ystart + 1 ) : ( yend + 1 ) : 2 , ( xstart + 1 ) : ( xend + 1) : 2 ]    # matrix of 3x3 blue Top Right pixels


      Lambda = 1

      img_out[y    , x     , R] = red[1][1]
      img_out[y    , x     , B] = img_green[y    , x    ]  + Lambda*( chromaBlue[0][0] + chromaBlue[0][1] + chromaBlue[1][1] + chromaBlue[1][0] ) / 4
      img_out[y    , x     , G] = img_green[y    , x    ]

      img_out[y + 1, x     , R] = img_green[y + 1, x    ]  + Lambda*( chromaRed[1][1]  + chromaRed[2][1]  ) / 2
      img_out[y + 1, x     , B] = img_green[y + 1, x    ]  + Lambda*( chromaBlue[1][0] + chromaBlue[1][1] ) / 2
      img_out[y + 1, x     , G] = img_green[y + 1, x    ]

      img_out[y + 1, x + 1 , R] = img_green[y + 1, x + 1]  + Lambda*( chromaRed[1][1]  + chromaRed[1][2] +  chromaRed[2][1] + chromaRed[2][2]   ) / 4
      img_out[y + 1, x + 1 , B] = blue[1][1]
      img_out[y + 1, x + 1 , G] = img_green[y + 1, x + 1]

      img_out[y    , x + 1 , R] = img_green[y    , x + 1]  + Lambda*( chromaRed[1][1]  + chromaRed[1][2]  ) / 2
      img_out[y    , x + 1 , B] = img_green[y    , x + 1]  + Lambda*( chromaBlue[0][1] + chromaBlue[1][1] ) / 2
      img_out[y    , x + 1 , G] = img_green[y    , x + 1]

    # *************
  img_out[img_out < 0] = 0

  return img_out

# Applies the improved Demosaicking
def improvedDM(img_in) -> np.ndarray :
  #green_img = bilinearGreenUpsamplingDM(img_in)
  green_img = hamiltonAdamsGreenUpsamplingDM(img_in)
  print("AfterHA:",green_img.min(),green_img.max())

  out_img   = bilinearChromaGreenGuidedUpsamplingDM(img_in, green_img)
  print("AfterCU:",out_img.min(),out_img.max())

  return out_img

# Applies some Color Matrix on an RGB image
def ispCM(img_in, color_matrix) -> np.ndarray :
  h, w, c = img_in.shape
  img_out = img_in.copy().flatten().reshape((-1, 3))
  for index, pixel in enumerate(img_out):
    pixel = np.dot(color_matrix, pixel)
    img_out[index] = pixel
  img_out = img_out.reshape((h,w,3))
  img_out[img_out < 0] = 0
  return img_out

# Applies a standard sRGB Tone curve on an RGB image
def ispTC(img_in, whiteLevel) -> np.ndarray :
  h, w, c = img_in.shape
  img_out = img_in.copy().flatten()
  img_out[img_out < 0] = 0
  img_out = img_out/whiteLevel

  for index, val in enumerate(img_out):
    if val < 0.0031308:
        img_out[index] = 12.92*val
    else:	
        img_out[index] = 1.055*math.pow(val, 1/2.4) - 0.055
  img_out = img_out.reshape((h, w, 3))

  return img_out

def isp1(img_in, bayerPattern, blackLevel, gain, WBGains, colorMatrix, whiteLevel) -> np.ndarray :
  print("RAW:",img_in.min(), img_in.max())

  blc_raw = ispBL(img_in, blackLevel, bayerPattern)
  print("AfterBL:",blc_raw.min(), blc_raw.max())

  exp_raw = gain*blc_raw.copy()
  print("AfterEC:",exp_raw.min(), exp_raw.max())

  img_dms = ispBilinearDM(exp_raw)
  print("AfterDM:",img_dms.min(), img_dms.max())

  img_wb = ispWB(img_dms, WBGains)
  print("AfterWB:",img_wb.min(), img_wb.max())

  img_ccm = ispCM(img_wb, colorMatrix)
  print("AfterCM:",img_ccm.min(), img_ccm.max())

  img_out = ispTC(img_ccm, whiteLevel)
  print("AfterTC:",img_out.min(), img_out.max())  

  return img_out

def isp2(img_in, bayerPattern, blackLevel, gain, WBGains, colorMatrix, whiteLevel) -> np.ndarray :
  print("RAW:",img_in.min(), img_in.max())

  blc_raw = ispBL(img_in, blackLevel, bayerPattern)
  print("AfterBL:",blc_raw.min(), blc_raw.max())

  exp_raw = gain*blc_raw.copy()
  print("AfterEC:",exp_raw.min(), exp_raw.max())

  wb_raw = ispRawWB(exp_raw, WBGains)
  print("AfterWB:",wb_raw.min(), wb_raw.max())

  img_dms = improvedDM(wb_raw)
  print("AfterDM:",img_dms.min(), img_dms.max())

  img_ccm = ispCM(img_dms, colorMatrix)
  print("AfterCM:",img_dms.min(), img_dms.max())

  img_out = ispTC(img_ccm, whiteLevel)
  print("AfterTC:",img_out.min(), img_out.max())

  return img_out

if __name__ == '__main__':
    raw = rawpy.imread(r".\SonyA7S3\ISO100.dng")

    bp = raw.raw_pattern
    blc = raw.black_level_per_channel
    wl  = raw.white_level
    wb = np.array(raw.camera_whitebalance)
    cm = raw.color_matrix[0:3,0:3]
    gain = 2

    cropRaw = np.array(raw.raw_image.copy()).reshape((raw.sizes.raw_height, raw.sizes.raw_width)).astype('float')[900:1300,1950:2350]
    # cropRaw = np.array(raw.raw_image.copy()).reshape((raw.sizes.raw_height, raw.sizes.raw_width)).astype('float')

    cropIsp1 = isp1(cropRaw, bp, blc, 2, wb[:3], cm, wl)
    cropIsp2 = isp2(cropRaw, bp, blc, 2, wb[:3], cm, wl)

    outimg = cropIsp1.copy()
    outimg[outimg < 0] = 0
    outimg[outimg > 1] = 1
    outimg = outimg * 255
    imageio.imwrite("ISO100_ISP1.jpg", outimg.astype('uint8'))

    outimg = cropIsp2.copy()
    outimg[outimg < 0] = 0
    outimg[outimg > 1] = 1
    outimg = outimg * 255
    imageio.imwrite("ISO100_ISP2.jpg", outimg.astype('uint8'))
import numpy as np

from scipy import misc

# for image_path in glob.glob("/home/adam/*.png"):
#     image = misc.imread(image_path)
#     print image.shape
#     print image.dtype
img = misc.imread("imgs/forest.jpg")
idealArea = 400*400
actualArea = img.shape[0] * img.shape[1]
scaleFactor = idealArea/float(actualArea)
img = misc.imresize(img, scaleFactor)



def interpolate(img, x, y):
    ix = int(x)
    iy = int(y)
    fx = x - ix
    fy = y - iy
    p00 = img[iy, ix]
    p01 = img[iy, ix + 1]
    p10 = img[iy + 1, ix]
    p11 = img[iy + 1, ix + 1]
    return (1 - fy)*(1 - fx)*p00 + (1 - fy)*fx*p01 + fy*(1 - fx)*p10 + fy*fx*p11

def cropImg(img, cx, cy, winSize):
    assert winSize % 2 == 1
    border = int((winSize - 1) / 2)
    newImg = np.zeros((winSize, winSize, img.shape[2]))
    for y in range(-border, border + 1):
        for x in range(-border, border + 1):
            newImg[border + y, border + x] = interpolate(img, cx + x, cy + y)
    return newImg

def drawHatch(img, cx, cy, clr = [255, 0, 0], hatch_size=2):
    img[cx - hatch_size : cx + hatch_size + 1, cy] = np.array(clr)
    img[cx, cy - hatch_size : cy + hatch_size + 1] = np.array(clr)

def fillRect(img, x0, x1, y0, y1, c):
    img[y0:y1 + 1, x0:x1 + 1] = c

def drawRect(img, x0, x1, y0, y1, c = [255, 0, 0]):
    print x0, x1, y0, y1
    fillRect(img, x0, x0, y0, y1, c)
    fillRect(img, x1, x1, y0, y1, c)
    fillRect(img, x0, x1, y0, y0, c)
    fillRect(img, x0, x1, y1, y1, c)

# Note: scipy.misc.imresize has a wierd rounding problem. This makes
# it not as nice for using 'nearest' interpolation, so we created our
# own function
def scale(img, factor):
    newImg = np.zeros((img.shape[0]*factor, img.shape[1]*factor, img.shape[2]))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            newImg[y*factor:(y+1)*factor, x*factor:(x+1)*factor] = img[y, x]
    return newImg

def showLarge(img, factor = 9, draw = False):
    cx = int((img.shape[0]-1)/2) * factor
    cy = int((img.shape[1]-1)/2) * factor
    img = scale(img, factor)
    print img.shape
    #Draw cross hatch
    #drawHatch(img, cx + (factor-1)/2, cy + (factor-1)/2)
    drawRect(img, cx, cx + factor - 1, cy, cy + factor - 1)
    if draw:
        misc.imshow(img)
    else:
        return img

def combine(imgs, draw = True):
    img = np.concatenate(imgs, 1)
    if draw:
        misc.imshow(img)
    else:
        return img

x = int(250*scaleFactor)
y = int(300*scaleFactor)

img1 = cropImg(img, x, y, 21)
img2 = cropImg(img, x, y + .5, 21)
combine((showLarge(img1), showLarge(img2)))









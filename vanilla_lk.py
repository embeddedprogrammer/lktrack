import numpy as np
import cv2
#img = cv2.imread("imgs/bat.png", cv2.IMREAD_UNCHANGED)
img = cv2.imread("imgs/forest.jpg")
gray = img[:, :, 0]

#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




import numpy as np
from scipy import misc
from scipy import signal

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
    if len(img.shape) == 3:
        newImg = np.zeros((winSize, winSize, img.shape[2]))
    else:
        newImg = np.zeros((winSize, winSize))
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

# Note: scipy.misc.imresize has a weird rounding problem. This makes
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

def getGrad(img):
    scharr = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    gradx = signal.convolve2d(img, scharr, mode='same')
    grady = signal.convolve2d(img, scharr.T, mode='same')
    return gradx, grady

def testCV(gray, p0, p1, count = 1, maxLevel = 0, winSize = (21, 21)):
    lk_params = dict( winSize = winSize,
                      maxLevel = maxLevel,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, count, 1e-9),
                      flags = cv2.OPTFLOW_USE_INITIAL_FLOW)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray, gray, p0, p1, **lk_params)
    return p1

img = img[:, :, 0]

true_x = int(250*scaleFactor)
true_y = int(300*scaleFactor)
print "True x: %f, y: %f" %(true_x, true_y)

template = cropImg(img, true_x, true_y, 21)
x = true_x + 0.1
y = true_y + 0.1
print "Start x: %f, y: %f" %(x, y)

# precompute template gradient
dx, dy = getGrad(template)
J = np.concatenate((np.reshape(dx, (-1, 1)), np.reshape(dy, (-1, 1))), axis=1)
H_inv = np.linalg.inv(np.matmul(J.T, J))

for i in range(20):
    # compare with opencv
    p0 = np.array([[[true_x, true_y]]], dtype='float32')
    p1 = np.array([[[x, y]]]).astype(dtype='float32')
    p2 = testCV(img, p0, p1)
    cx = p2[0, 0, 0]
    cy = p2[0, 0, 1]

    # warp image (in this case it is only translating the image)
    croppedImg = cropImg(img, x, y, 21)
    r = np.reshape(croppedImg - template, (-1, 1))
    e = np.linalg.norm(r)
    #dp = np.linalg.lstsq(J, err_1d)[0]
    deltaX = np.matmul(H_inv, np.matmul(J.T, r))
    x += deltaX[0, 0]
    y += deltaX[1, 0]

    print "Iter %d err: %f x: %f, y: %f cx: %f, cy: %f" %(i, e, x, y, cx, cy)

#showLarge(diff, draw=True)
#combine((showLarge(img1), showLarge(img2)))










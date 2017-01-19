import numpy as np
import cv2

from scipy import misc
from scipy import signal


class LK:
    def __init__(self, img, x, y, method='custom', gradientMethod='scharr', winSize=21, count=10, err=1e-4, maxLevel=0):
        self.lastImg = img
        self.x = x
        self.y = y
        self.method = method
        self.gradientMethod = gradientMethod
        self.count = count
        self.err = err
        self.winSize = winSize
        self.maxLevel = maxLevel

    @staticmethod
    def interpolate(img, x, y):
        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy
        p00 = img[iy, ix]
        p01 = img[iy, ix + 1]
        p10 = img[iy + 1, ix]
        p11 = img[iy + 1, ix + 1]
        return (1 - fy) * (1 - fx) * p00 + (1 - fy) * fx * p01 + fy * (1 - fx) * p10 + fy * fx * p11

    @staticmethod
    def cropImg(img, cx, cy, winSize, fullSize=False):
        assert winSize % 2 == 1
        border = int((winSize - 1) / 2)
        if fullSize:
            newImg = np.zeros_like(img)
        elif len(img.shape) == 3:
            newImg = np.zeros((winSize, winSize, img.shape[2]))
        else:
            newImg = np.zeros((winSize, winSize))
        for y in range(-border, border + 1):
            for x in range(-border, border + 1):
                newImg[border + y, border + x] = LK.interpolate(img, cx + x, cy + y)
        return newImg

    @staticmethod
    def drawHatch(img, cx, cy, clr=[0, 0, 255], hatch_size=2):
        p0 = np.array([cx, cy], dtype=np.float32)
        p1 = np.array([cx + 10, cy], dtype=np.float32)
        cv2.line(img, tuple(p0), tuple(p1), (255, 0, 0))
        img[cy - hatch_size: cy + hatch_size + 1, cx] = np.array(clr)
        img[cy, cx - hatch_size: cx + hatch_size + 1] = np.array(clr)

    @staticmethod
    def fillRect(img, x0, x1, y0, y1, c):
        img[y0:y1 + 1, x0:x1 + 1] = c

    @staticmethod
    def drawRect(img, x0, x1, y0, y1, c=[255, 0, 0]):
        print x0, x1, y0, y1
        LK.fillRect(img, x0, x0, y0, y1, c)
        LK.fillRect(img, x1, x1, y0, y1, c)
        LK.fillRect(img, x0, x1, y0, y0, c)
        LK.fillRect(img, x0, x1, y1, y1, c)

    # Note: scipy.misc.imresize has a weird rounding problem. This makes
    # it not as nice for using 'nearest' interpolation, so we created our
    # own function
    @staticmethod
    def scale(img, factor):
        newImg = np.zeros((img.shape[0] * factor, img.shape[1] * factor, img.shape[2]))
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                newImg[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor] = img[y, x]
        return newImg

    @staticmethod
    def showLarge(img, factor=9, draw=False):
        cx = int((img.shape[0] - 1) / 2) * factor
        cy = int((img.shape[1] - 1) / 2) * factor
        img = LK.scale(img, factor)
        print img.shape
        # Draw cross hatch
        # drawHatch(img, cx + (factor-1)/2, cy + (factor-1)/2)
        LK.drawRect(img, cx, cx + factor - 1, cy, cy + factor - 1)
        if draw:
            misc.imshow(img)
        else:
            return img

    @staticmethod
    def combine(imgs, draw=True):
        img = np.concatenate(imgs, 1)
        if draw:
            misc.imshow(img)
        else:
            return img

    @staticmethod
    def getGradient(img, method='cv_scharr'):
        if method == 'cv_scharr':
            gradx = cv2.Scharr(img, cv2.CV_64F, 1, 0) / 32
            grady = cv2.Scharr(img, cv2.CV_64F, 0, 1) / 32
            return gradx, grady

        if method == 'simple':
            kernel = np.array([[-1, 0, 1]], dtype=np.float32) / 2
        elif method == 'sobel':
            kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32) / 8
        elif method == 'scharr':
            kernel = np.array([[-3, 0, 3],
                               [-10, 0, 10],
                               [-3, 0, 3]], dtype=np.float32) / 32

        gradx = cv2.filter2D(img, -1, kernel)
        grady = cv2.filter2D(img, -1, kernel.T)
        return gradx, grady

    def track(self, img):
        if self.lastImg is None:
            self.lastImg = img
            return

        if self.method == 'custom':
            # Extract template image patch
            template = LK.cropImg(self.lastImg, self.x, self.y, self.winSize)
            dx, dy = LK.getGradient(template, method=self.gradientMethod)

            # Precompute jacobian and inverse of grammian (these are constant)
            J = np.concatenate((np.reshape(dx, (-1, 1)), np.reshape(dy, (-1, 1))), axis=1)
            G_inv = np.linalg.inv(np.matmul(J.T, J))

            # Iterative gauss-newton algorithm
            for i in range(self.count):
                # Extract patch from current image and compute error
                croppedImg = LK.cropImg(img, self.x, self.y, 21)
                residual = np.reshape(template - croppedImg, (-1, 1))
                error = np.linalg.norm(residual)
                if error < self.err:
                    print "Iter %d Terminate at err: %.3f" %(i, error)
                    break

                # Compute least-squares solution to linearized equation
                deltaX = np.matmul(G_inv, np.matmul(J.T, residual))
                self.x += deltaX[0, 0]
                self.y += deltaX[1, 0]
                print "Iter %d err: %.3f   lk: (%.3f, %.3f)" % (i, error, self.x, self.y)

        elif self.method == 'opencv':
            lk_params = dict(winSize=(self.winSize, self.winSize),
                             maxLevel=self.maxLevel,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.count, self.err))
            p0 = np.array([[[self.x, self.y]]], dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.lastImg, img, p0, **lk_params)
            self.x = p1[0, 0, 0]
            self.y = p1[0, 0, 1]

        self.lastImg = img
        return self.x, self.y
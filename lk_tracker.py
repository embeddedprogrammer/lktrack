import numpy as np
import cv2

from scipy import misc
from scipy import signal


class LK:
    def __init__(self, img, fg_mask, x, y, method='custom', gradientMethod='scharr', winSize=21, count=10, err=1e-4, maxLevel=0, weights=None, thresh=None):
        self.lastImg = img
        self.lastMask = fg_mask
        self.x = x
        self.y = y
        self.method = method
        self.gradientMethod = gradientMethod
        self.count = count
        self.err = err
        self.winSize = winSize
        self.maxLevel = maxLevel
        self.weights = weights
        self.thresh = thresh

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
        pt = np.array([cx, cy], dtype=np.float32)
        down = np.array([hatch_size, 0.], dtype=np.float32)
        right = np.array([0., hatch_size], dtype=np.float32)
        cv2.line(img, tuple(pt - down), tuple(pt + down), clr, lineType=cv2.CV_AA)
        cv2.line(img, tuple(pt + right), tuple(pt - right), clr, lineType=cv2.CV_AA)

    @staticmethod
    def drawSquare(img, cx, cy, clr=[0, 0, 255], winsize=21):
        pt = np.array([cx, cy], dtype=np.float32)
        down = np.array([(winsize - 1) / 2, 0.], dtype=np.float32)
        right = np.array([0., (winsize - 1) / 2], dtype=np.float32)
        cv2.rectangle(img, tuple(pt - down - right), tuple(pt + down + right), clr, lineType=cv2.CV_AA)

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

    @staticmethod
    def createThreshold(bg, fg):
        bg_av = np.mean(np.mean(bg[:, :, :3], axis=0), axis=0)
        fg_av = np.mean(np.mean(fg[:, :, :3], axis=0), axis=0)
        overall_av = (fg_av + bg_av) / 2
        gradient = fg_av - bg_av
        weights = gradient.astype(dtype=np.float32) / np.sum(np.abs(gradient))
        thresh = np.sum(overall_av * weights)
        return weights, thresh

    @staticmethod
    def divThreshold(img, weights):
        gray = np.sum(img * weights, axis=2)
        gray2 = cv2.normalize(gray, norm_type=cv2.NORM_MINMAX)
        return gray2

    @staticmethod
    def boolThreshold(img, weights, thresh):
        gray = np.sum(img * weights, axis=2)
        retval, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return bw

    @staticmethod
    def maskThreshold(img, weights, thresh):
        gray = np.sum(img * weights, axis=2)
        return np.expand_dims(gray > thresh, axis=2)

    def track(self, img, fg_mask):
        if self.lastImg is None:
            self.lastImg = img
            return

        if self.method == 'lk':
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
                    #print "Iter %d Terminate at err: %.3f" %(i, error)
                    break

                # Compute least-squares solution to linearized equation
                deltaX = np.matmul(G_inv, np.matmul(J.T, residual))
                self.x += deltaX[0, 0]
                self.y += deltaX[1, 0]
                #print "Iter %d err: %.3f   lk: (%.3f, %.3f)" % (i, error, self.x, self.y)

        elif self.method == 'lk_mask':
            # Extract template image patch
            template = LK.cropImg(self.lastImg, self.x, self.y, self.winSize)
            mask =    LK.cropImg(self.lastMask, self.x, self.y, self.winSize)
            #mask = LK.maskThreshold(template, self.weights, self.thresh)
            dx, dy = LK.getGradient(template, method=self.gradientMethod)
            dx *= mask
            dy *= mask

            # Precompute jacobian and inverse of grammian (these are constant)
            J = np.concatenate((np.reshape(dx, (-1, 1)), np.reshape(dy, (-1, 1))), axis=1)
            G_inv = np.linalg.inv(np.matmul(J.T, J))

            # Iterative gauss-newton algorithm
            for i in range(self.count):
                # Extract patch from current image and compute error
                croppedImg = LK.cropImg(img, self.x, self.y, self.winSize)
                mask =   LK.cropImg(fg_mask, self.x, self.y, self.winSize)
                #mask = LK.maskThreshold(croppedImg, self.weights, self.thresh)
                diff_masked = (template - croppedImg) * mask
                if i == 0:
                    cv2.imshow('mask', mask.astype(dtype=np.uint8)*255)

                residual = np.reshape(diff_masked, (-1, 1))
                error = np.linalg.norm(residual)
                if error < self.err:
                    #print "Iter %d Terminate at err: %.3f" %(i, error)
                    break

                # Compute least-squares solution to linearized equation
                deltaX = np.matmul(G_inv, np.matmul(J.T, residual))
                self.x += deltaX[0, 0]
                self.y += deltaX[1, 0]
                print "Iter %d err: %.3f   lk: (%.3f, %.3f)" % (i, error, self.x, self.y)

        elif self.method == 'lk_mask2':
            # Extract template image patch
            template = LK.cropImg(self.lastImg, self.x, self.y, self.winSize)
            mask =    LK.cropImg(self.lastMask, self.x, self.y, self.winSize)
            # mask = LK.maskThreshold(template, self.weights, self.thresh)
            dx, dy = LK.getGradient(template, method=self.gradientMethod)
            dx *= mask
            dy *= mask

            # Precompute jacobian and inverse of grammian (these are constant)
            J = np.concatenate((np.reshape(dx, (-1, 1)), np.reshape(dy, (-1, 1))), axis=1)
            G_inv = np.linalg.inv(np.matmul(J.T, J))

            # Iterative gauss-newton algorithm
            for i in range(self.count):
                # Extract patch from current image and compute error
                croppedImg = LK.cropImg(img, self.x, self.y, 21)
                #mask = LK.maskThreshold(croppedImg, self.weights, self.thresh) #Faster to use template mask
                diff_masked = (template - croppedImg) * mask
                if i == 0:
                    cv2.imshow('mask', mask.astype(dtype=np.uint8)*255)

                residual = np.reshape(diff_masked, (-1, 1))
                error = np.linalg.norm(residual)
                if error < self.err:
                    #print "Iter %d Terminate at err: %.3f" %(i, error)
                    break

                # Compute least-squares solution to linearized equation
                deltaX = np.matmul(G_inv, np.matmul(J.T, residual))
                self.x += deltaX[0, 0]
                self.y += deltaX[1, 0]
                print "Iter %d err: %.3f   lk: (%.3f, %.3f)" % (i, error, self.x, self.y)

        elif self.method == 'lk_mask3':
            # Extract template image patch
            template = LK.cropImg(self.lastImg, self.x, self.y, self.winSize)
            mask    = LK.cropImg(self.lastMask, self.x, self.y, self.winSize)
            #mask = LK.maskThreshold(template, self.weights, self.thresh)

            # Iterative gauss-newton algorithm
            for i in range(self.count):
                # Extract patch from current image
                croppedImg = LK.cropImg(img, self.x, self.y, 21)

                # Compute gradient of image
                dx, dy = LK.getGradient(croppedImg, method=self.gradientMethod)
                dx *= mask
                dy *= mask

                # Compute Jacobian and inverse of grammian
                J = np.concatenate((np.reshape(dx, (-1, 1)), np.reshape(dy, (-1, 1))), axis=1)
                G_inv = np.linalg.inv(np.matmul(J.T, J))

                # Compute error (masked)
                diff_masked = (template - croppedImg) * mask
                if i == 0:
                    cv2.imshow('mask', mask.astype(dtype=np.uint8)*255)

                residual = np.reshape(diff_masked, (-1, 1))
                error = np.linalg.norm(residual)
                if error < self.err:
                    #print "Iter %d Terminate at err: %.3f" %(i, error)
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
        self.lastMask = fg_mask
        return self.x, self.y
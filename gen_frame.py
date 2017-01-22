import numpy as np
import cv2
import time
from lk_tracker import LK
import math
import matplotlib.pyplot as plt

key_esc = 27
key_esc = 27
key_left = 81
key_up = 82
key_right = 83
key_down = 84

global times
times = []

def tic():
    times.append(time.time())

def toc(name = ""):
    if len(times) == 0:
        print "Must call tic before toc"
    else:
        t = time.time() - times.pop()
        print "%s: %.0f ms" %(name, t * 1000)

def overlayImg(bg, fg):
    alpha = fg[:, :, 3:4] / 255.
    return ((1 - alpha)*bg[:, :, :3] + alpha*fg[:, :, :3]).astype(np.uint8)

def overlay2(bg, fg, pt_fg, pt_bg, scale):
    down = np.array([[1., 0.]], dtype=np.float32)
    right = np.array([[0., 1.]], dtype=np.float32)
    pts_fg = np.stack([pt_fg, pt_fg + down, pt_fg + right + down, pt_fg + right], axis=0)
    down *= scale
    right *= scale
    pts_bg = np.stack([pt_bg, pt_bg + down, pt_bg + right + down, pt_bg + right], axis=0)
    T = cv2.getPerspectiveTransform(pts_fg, pts_bg)
    fg_warped = cv2.warpPerspective(fg, T, (bg.shape[1], bg.shape[0]))
    img = overlayImg(bg, fg_warped)
    fg_mask = fg_warped[:, :, 3:4]
    return img, fg_mask

def genTrackAndShow(lktracker, x, y, size, method, weights, thresh, showFrame=True):
    # generate next frame
    img, fg_mask = overlay2(bg, fg, np.array([[fg.shape[1] / 2, fg.shape[0] / 2]], dtype=np.float32), np.array([[x, y]], dtype=np.float32), scale=size/fg.shape[0])

    # track
    if lktracker is None:
        lktracker = LK(img, fg_mask, x, y, method=method, weights=weights, thresh=thresh)
    else:
        lktracker.track(img, fg_mask)

        # display image
        if showFrame:
            disp = np.copy(img)
            #lktracker.drawHatch(disp, lktracker.x, lktracker.y, hatch_size=20)
            lktracker.drawSquare(disp, lktracker.x, lktracker.y, winsize=lktracker.winSize)
            cv2.imshow('image', disp)

    return lktracker

def playWithTracker(fg, bg, method):
    weights, thresh = LK.createThreshold(bg, fg)

    x = 100.
    y = 100.
    dist = 1.
    size = 20.
    lktracker = genTrackAndShow(None, x, y, size, method, weights, thresh)

    while True:
        key = cv2.waitKey(0) & 0xff
        if key == key_esc or key == ord('q'):
            break
        elif key == key_left:
            x -= dist
        elif key == key_right:
            x += dist
        elif key == key_up:
            y -= dist
        elif key == key_down:
            y += dist
        elif key == ord('+'):
            size += 1
        elif key == ord('-'):
            size -= 1
        else:
            print key
        lktracker = genTrackAndShow(lktracker, x, y, size, method, weights, thresh)
    cv2.destroyAllWindows()

def testTracker(fg, bg, method):
    #weights, thresh = LK.createThreshold(bg, fg)
    weights, thresh = (None, None)
    size = 20.
    lktracker = None
    cx = bg.shape[1] / 2
    cy = bg.shape[0] / 2
    radius = min(cx, cy) * 0.8
    dist = 3. #2.
    omega = dist/radius
    iters = 100
    err = np.zeros(iters, dtype=np.float32)
    times = np.arange(iters)
    for t in range(iters):
        x = cx + math.cos(omega*t) * radius
        y = cy + math.sin(omega*t) * radius
        lktracker = genTrackAndShow(lktracker, x, y, size, method, weights, thresh)
        dx = lktracker.x - x
        dy = lktracker.y - y
        err[t] = math.sqrt(dx*dx + dy*dy)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    return err, times

def testTrackers():
    methods = ['opencv', 'lk_mask', 'lk_mask2', 'lk_mask3']
    for method in methods:
        err, times = testTracker(fg, bg, method)
        plt.plot(times, err)
    plt.legend(methods, loc='lower right')
    plt.xlabel('Time')
    plt.ylabel('Error (pixels)')
    plt.show()

# Load images
#bg = cv2.imread("imgs/forest.jpg")
#bg = cv2.imread("imgs/chess.jpg")
bg = cv2.imread("imgs/noise.png")
fg = cv2.imread("imgs/bat.png", cv2.IMREAD_UNCHANGED)

ideal_width = 300.
f_bg = ideal_width / bg.shape[1]
fg = cv2.resize(fg, (0, 0), fx=.3, fy=.3)
bg = cv2.resize(bg, (0, 0), fx=f_bg, fy=f_bg)

#playWithTracker(fg, bg, 'lk_mask3')
testTrackers()


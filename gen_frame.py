import numpy as np
import cv2
import time
from lk_tracker import LK

key_esc = 27
key_esc = 27
key_left = 81
key_up = 82
key_right = 83
key_down = 84

# Load images
bg = cv2.imread("imgs/forest.jpg")
fg = cv2.imread("imgs/bat.png", cv2.IMREAD_UNCHANGED)

fg = cv2.resize(fg, (0, 0), fx=.3, fy=.3)
bg = cv2.resize(bg, (0, 0), fx=.3, fy=.3)

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
    return img

def genTrackAndShow(lktracker, x, y, size):
    # generate next frame
    img = overlay2(bg, fg, np.array([[fg.shape[1] / 2, fg.shape[0] / 2]], dtype=np.float32), np.array([[x, y]], dtype=np.float32), scale=size/fg.shape[0])

    # track
    if lktracker is None:
        lktracker = LK(img, x, y, method='opencv')
    else:
        lktracker.track(img)

        # display image
        disp = np.copy(img)
        lktracker.drawHatch(disp, lktracker.x, lktracker.y, hatch_size=20)
        cv2.imshow('image', disp)
    return lktracker

x = 100.
y = 100.
dist = 1.
size = 20.
lktracker = genTrackAndShow(None, x, y, size)

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
    lktracker = genTrackAndShow(lktracker, x, y, size)

cv2.destroyAllWindows()



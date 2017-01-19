import numpy as np
import cv2
import time

key_esc = 27
key_esc = 27
key_left = 81
key_up = 82
key_right = 83
key_down = 84

# Load images
bg = cv2.imread("imgs/forest.jpg")
fg = cv2.imread("imgs/bat.png", cv2.IMREAD_UNCHANGED)
#bg = bg[:275, :494, :3]

def tic():
    global t0
    t0 = time.time()

def toc(name = ""):
    t = time.time() - t0
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
    return overlayImg(bg, fg_warped)

x = 100.
y = 100.
dist = 10.

while True:
    key = cv2.waitKey(-1) & 0xff
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
    else:
        print key
    tic()
    img = overlay2(bg, fg, np.array([[fg.shape[1] / 2, fg.shape[0] / 2]], dtype=np.float32), np.array([[x, y]], dtype=np.float32), scale=.1)
    toc("Total")
    cv2.imshow('image', img)

cv2.destroyAllWindows()



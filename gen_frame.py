import numpy as np
import cv2

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

def overlayImg(bg, fg):
    alpha = fg[:, :, 3:4] / 255.
    return ((1 - alpha)*bg[:, :, :3] + alpha*fg[:, :, :3]).astype(np.uint8)

def overlay2(bg, fg, pt_fg, pt_bg):
    down = np.array([[1., 0.]], dtype=np.float32)
    right = np.array([[0., 1.]], dtype=np.float32)
    pts_bg = np.stack([pt_bg, pt_bg + down, pt_bg + right + down, pt_bg + right], axis=0)
    pts_fg = np.stack([pt_fg, pt_fg + down, pt_fg + right + down, pt_fg + right], axis=0)
    T = cv2.getPerspectiveTransform(pts_fg, pts_bg)
    fg_warped = cv2.warpPerspective(fg, T, (bg.shape[1], bg.shape[0]))
    return overlayImg(bg, fg_warped)

img = overlay2(bg, fg, np.array([[0., 0.]], dtype=np.float32), np.array([[100., 0.]], dtype=np.float32))

cv2.imshow('bg', bg)
cv2.imshow('fg', fg)
cv2.imshow('image', img)
while True:
    key = cv2.waitKey(0) & 0xff
    if key == key_esc or key == ord('q'):
        break
    else:
        print key
cv2.destroyAllWindows()



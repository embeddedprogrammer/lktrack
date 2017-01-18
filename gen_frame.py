import numpy as np
import cv2

key_esc = 27
key_esc = 27
key_left = 81
key_up = 82
key_right = 83
key_down = 84

# Load images
img_bg = cv2.imread("imgs/forest.jpg")
img_fg = cv2.imread("imgs/bat.png", cv2.IMREAD_UNCHANGED)
img_bg = img_bg[:275, :494, :3]

def overlayImg(bg, fg):
    alpha = fg[:, :, 3:4] / 255.
    return ((1 - alpha)*bg[:, :, :3] + alpha*fg[:, :, :3]).astype(np.uint8)

img = overlayImg(img_bg, img_fg)
cv2.imshow('bg', img_bg)
cv2.imshow('fg', img_fg)
cv2.imshow('image', img)
while True:
    key = cv2.waitKey(0) & 0xff
    if key == key_esc:
        break
    else:
        print key
cv2.destroyAllWindows()



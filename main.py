import cv2
import numpy

print(cv2.__version__)
img = cv2.imread("a.png",cv2.IMREAD_COLOR)
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
cv2.namedWindow("Window",cv2.WINDOW_AUTOSIZE)
width = img.shape[1]
height = img.shape[0]

# img[0:height//2,width//2:width,1] = 0
# img[0:height//2,width//2:width,2] = 0
# img[height//2:height,0:width//2,0] = 0
# img[height//2:height,0:width//2,2] = 0
# img[height//2:height,width//2:width,0] = 0
# img[height//2:height,width//2:width,1] = 0

img[0:height//2,width//2:width,1:3] = 0
img[height//2:height,0:width//2,0:3:2] = 0
img[height//2:height,width//2:width,0:2] = 0


cv2.imshow("Window",img)

print(img.shape)
cv2.waitKey()


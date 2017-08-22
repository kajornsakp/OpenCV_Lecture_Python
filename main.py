import cv2
import numpy

# week1
# print(cv2.__version__)
# img = cv2.imread("a.png",cv2.IMREAD_COLOR)
# img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
# cv2.namedWindow("Window",cv2.WINDOW_AUTOSIZE)
# width = img.shape[1]
# height = img.shape[0]
#
# # img[0:height//2,width//2:width,1] = 0
# # img[0:height//2,width//2:width,2] = 0
# # img[height//2:height,0:width//2,0] = 0
# # img[height//2:height,0:width//2,2] = 0
# # img[height//2:height,width//2:width,0] = 0
# # img[height//2:height,width//2:width,1] = 0
#
# img[0:height//2,width//2:width,1:3] = 0
# img[height//2:height,0:width//2,0:3:2] = 0
# img[height//2:height,width//2:width,0:2] = 0
#
#
# cv2.imshow("Window",img)
#
# print(img.shape)
# cv2.waitKey()



#week 2
#
# img = cv2.imread("a.png")
# img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
# out = numpy.zeros((img.shape[0],img.shape[1])).astype(numpy.uint8)
# cv2.namedWindow("window",cv2.WINDOW_AUTOSIZE)
# blue = img[::,0]
# green = img[::,1]
# red = img[::,2]
# for i in range(0,img.shape[0]):
#     for j in range(0,img.shape[1]):
#         out[(i,j)] = (0.114*img[(i,j)][0]+0.587*img[(i,j)][1]+0.299*img[(i,j)][2])
#
#
#
# cv2.imshow("Window",out)
# cv2.waitKey()


# lut = numpy.ndarray(256,numpy.uint8)
# for i in range(lut.size):
#     lut[i] = 255-i;
#
# img = cv2.imread("a.png")
# img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
# out = cv2.LUT(img,lut)
# cv2.imshow("window",out)
# cv2.waitKey()



#histogram
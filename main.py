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



# week 2
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



# histogram
# def getHist(img):
#     hist = numpy.zeros(256,numpy.uint32)
#     rows,cols = img.shape
#     for r in range(rows):
#         for c in range(cols):
#             hist[img[r,c]] += 1
#     return hist
#
# def drawHist(hist):
#     histImg = numpy.full((256,256),255,numpy.uint8)
#     max = hist.max()
#     for z in range(256):
#         height = int(hist[z]*256/max)
#         cv2.line(histImg,(z,256),(z,256-height),0)
#     return histImg
#
# img = cv2.imread("a.png",cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)
# cv2.imshow("input",img)
# cv2.imshow("output",drawHist(getHist(img)))
# cv2.waitKey()


def contrastStretchLUT(pt1, pt2):
    lut = numpy.ndarray(256, numpy.uint8)
    m1 = pt1[1] / pt1[0]
    m2 = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    m3 = (255 - pt2[1]) / (255 - pt2[0])
    i = 0
    while i < pt1[0]:
        lut[i] = int(m1 * i)
    i += 1
    while i < pt2[0]:
        lut[i] = int(m2 * (i - pt1[0]) + pt1[1])
    i += 1
    while i < 256:
        lut[i] = int(m3 * (i - pt2[0]) + pt2[1])
    i += 1
    return lut


img = cv2.imread("a.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(0,0),fx=0.2,fy=0.2)

# lut = contrastStretchLUT((80,40), (175,215))
# outImg = cv2.LUT(img, lut)
outImg = cv2.equalizeHist(img)
cv2.imshow("Input image", img)
cv2.imshow("Processed image", outImg)
cv2.waitKey()
cv2.destroyAllWindows()

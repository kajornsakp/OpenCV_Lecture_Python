from itertools import chain

import cv2
import numpy as np


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


# img = cv2.imread("b.jpg", cv2.IMREAD_COLOR)
# img = cv2.resize(img,(0,0),fx=0.3,fy=0.3)

# lut = contrastStretchLUT((80,40), (175,215))
# outImg = cv2.LUT(img, lut)
# outImg = cv2.equalizeHist(img)
# outImg = cv2.GaussianBlur(img,(5,5),0)
#
# cv2.imshow("Input image", img)
# cv2.imshow("Processed image", outImg)
# cv2.waitKey()
#
# cv2.destroyAllWindows()


# for s in range(3,31,2):
#     print("filter size = ",s)
#     outImg = cv2.boxFilter(img,-1,(s,s))
#     cv2.imshow("processed image",outImg)
#     cv2.waitKey()

# cv2.imshow("Input image", img)
#
# for s in range(3,31,2):
#     print("filter size = ",s)
#     outImg = cv2.GaussianBlur(img,(s,s),0)
#     cv2.imshow("processed image",outImg)
#     cv2.waitKey()

#
# for sig in range(1,9):
#     print("sigma value = ",sig)
#     outImg = cv2.GaussianBlur(img,(31,31),sig)
#     cv2.imshow("processed image",outImg)
#     cv2.waitKey()


# img = cv2.imread("b.jpg", cv2.IMREAD_COLOR)
# # img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
# outImg = cv2.Laplacian(img,cv2.CV_16S,3)
# cv2.imshow("Input image",img)
# cv2.imshow("Output image",outImg)
# # print(outImg)
# outImg += 128
# # print(outImg)
# outImg = np.uint8(outImg)
# cv2.imshow("Add 128",outImg)
# cv2.waitKey()
#
# print(cv2.__version__)

#
# img = cv2.imread("b.jpg", cv2.IMREAD_GRAYSCALE)
# highpass = np.ones((3,3),np.uint8)*-1
# highpass[1,1] = 8
# print(highpass)
# outImg = cv2.filter2D(img,cv2.CV_8U,highpass)
# shpImg = cv2.addWeighted(img, 0.8, outImg, 0.2, 0)
# cv2.imshow("in",img)
# cv2.imshow("out",outImg)
# cv2.imshow("sharp",shpImg)
# cv2.waitKey()


# img = cv2.imread("b.jpg", cv2.IMREAD_COLOR)
# outImg = cv2.bilateralFilter(img,11,200,30)
# cv2.imshow("input",img)
# cv2.imshow("output",outImg)
# cv2.waitKey()


#
# def LoGFilter(img, ksize):
#     print(img)
#     blurImg = cv2.GaussianBlur(img, ksize, 0)
#     print("blur")
#     print(blurImg)
#     logImg = cv2.Laplacian(blurImg, cv2.CV_16S, 3)
#     print(logImg)
#     return logImg
#
#
# def zeroCrossing(img):
#     cv2.imshow("in image", img)
#     print(img)
#     outImg = np.zeros(img.shape, np.uint8)
#     TH = 5
#     rows, cols = img.shape
#     for r in range(1, rows - 1):
#         for c in range(1, cols - 1):
#             n = img[r - 1, c]
#             s = img[r + 1, c]
#             e = img[r, c + 1]
#             w = img[r, c - 1]
#             ne = img[r - 1, c + 1]
#             nw = img[r - 1, c - 1]
#             se = img[r + 1, c + 1]
#             sw = img[r + 1, c - 1]
#             if ((n * s) < 0 and abs(n - s) > TH) or \
#                     ((e * w) < 0 and abs(e - w) > TH) or \
#                     ((ne * sw) < 0 and abs(ne - sw) > TH) or \
#                     ((nw * se) < 0 and abs(nw - se) > TH):
#                 outImg[r, c] = 255
#     return outImg
#
#
# img = cv2.imread("b.jpg", cv2.IMREAD_GRAYSCALE)
# # img = cv2.resize(img,(0,0),fx=0.3,fy=0.3)
#
# logImg = LoGFilter(img, (9, 9))
# edgeImg = zeroCrossing(logImg)
# cv2.imshow("log",logImg)
# cv2.imshow("Input image", img)
# cv2.imshow("Edge image", edgeImg)
# cv2.waitKey()
# cv2.destroyAllWindows()

#
# img = cv2.imread("1001.jpg", cv2.IMREAD_GRAYSCALE)
# blurImg = cv2.GaussianBlur(img, (9,9), 0)
# edgeImg = cv2.Canny(blurImg, 50, 100)
# cv2.imshow("Input image", img)
# cv2.imshow("Edge image", edgeImg)
# cv2.waitKey()
# cv2.destroyAllWindows()


winName = "Canny"
img = cv2.imread("1001.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

def CannyThreshold(lth):
    blurImg = cv2.GaussianBlur(img, (9,9), 0)
    edgeImg = cv2.Canny(blurImg, lth, 2 * lth)
    cv2.imshow(winName, edgeImg)

lowTh = 0
maxLowTh = 100
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Min threshold", winName, lowTh, maxLowTh, CannyThreshold)
CannyThreshold(lowTh)
cv2.waitKey()
cv2.destroyAllWindows()
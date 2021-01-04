import cv2 as cv
import numpy
import matplotlib as np
from scipy.spatial import distance as dist

img_color = cv.imread("A1.jpg")
img_eg = cv.Canny(img_color, 100 ,200)
blur = cv.bilateralFilter(img_color,9,75,75)
img_gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 163, 255, cv.THRESH_BINARY)
img_binary = cv.bitwise_not(img_binary)
contours,hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)

# Contour 영역 내에 텍스트 쓰기
# https://github.com/bsdnoobz/opencv-code/blob/master/shape-detect.cpp
def setLabel(image, str, contour):

   fontface = cv.FONT_HERSHEY_SIMPLEX
   scale = 0.6
   thickness = 1

   size = cv.getTextSize(str, fontface, scale, thickness)
   text_width = size[0][0]
   text_height = size[0][1]

   x, y, width, height = cv.boundingRect(contour)

   pt = (x + int((width - text_width) / 2), y + int((height + text_height) / 2))
   cv.putText(image, str, pt, fontface, scale, (255, 255, 255), thickness, 1)

def label(image, contour):


   mask = numpy.zeros(image.shape[:2], dtype="uint8")
   cv.drawContours(mask, [contour], -1, 255, -1)

   mask = cv.erode(mask, None, iterations=2)
   mean = cv.mean(image, mask=mask)[:3]


   minDist = (numpy.inf, None)



   for (i, row) in enumerate(lab):

       d = dist.euclidean(row[0], mean)

       if d < minDist[0]:
           minDist = (d, i)

   return colorNames[minDist[1]]

# 인식할 색 입력
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
colorNames = ["red", "green", "blue"]

lab = numpy.zeros((len(colors), 1, 3), dtype="uint8")
for i in range(len(colors)):
   lab[i] = colors[i]

img_lab = cv.cvtColor(blur, cv.COLOR_BGR2LAB)

thresh = cv.erode(img_binary, None, iterations=2)

# 컨투어 리스트가 OpenCV 버전에 따라 차이있기 때문에 추가
if len(contours) == 2:
   contours = contours[0]

elif len(contours) == 3:
   contours = contours[1]


# 컨투어 별로 체크
for contour in contours:
   # 컨투어를 그림
   cv.drawContours(img_color, [contour], -1, (0, 255, 0), 2)

   # 컨투어 내부에 검출된 색을 표시
   color_text = label(img_lab, contour)
   setLabel(img_color, color_text, contour)


cv.imshow("result",img_color)
cv.waitKey(0)
import cv2 as cv
import numpy
import matplotlib as np
from scipy.spatial import distance as dist

img_color = cv.imread("A1.jpg")
img_eg = cv.Canny(img_color, 100 ,200)
blur = cv.bilateralFilter(img_color,9,100,100)
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
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255],[129,193,71],[255,212,0],[255,255,255],[255,51,153],[150,75,0],[0,86,102],[255,0,255],[139,0,255],[0,0,0],[0,0,128]]
colorNames = ["red", "green", "blue","light green","yellow","white","pink","brown","green blue","magenta","purple","black","dark blue"]

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
   color_text = label(img_color, contour)
   setLabel(img_color, color_text, contour)

for cnt in contours:
   
    epsilon = 0.005 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    size = len(approx)
    print(size)

    cv.line(img_color, tuple(approx[0][0]), tuple(approx[size-1][0]), (0, 255, 0), 3)
    for k in range(size-1):
        cv.line(img_color, tuple(approx[k][0]), tuple(approx[k+1][0]), (0, 255, 0), 3)

    if cv.isContourConvex(approx):
        if size == 3:
            setLabel(img_color, "triangle", cnt)
        elif size == 4:
            setLabel(img_color, "rectangle", cnt)
        elif size == 5:
            setLabel(img_color, "pentagon", cnt)
        elif size == 6:
            setLabel(img_color, "hexagon", cnt)
        elif size == 8:
            setLabel(img_color, "octagon", cnt)
        elif size == 10:
            setLabel(img_color, "decagon", cnt)
        else:
            setLabel(img_color, str(size), cnt)
    else:
        setLabel(img_color, str(size), cnt)


re1 = cv.resize(img_color,dsize=(1920,1080))
re2 = cv.resize(blur,dsize=(1920,1080))
cv.imshow("result",re1)
cv.imshow("blur",re2)
cv.waitKey(0)
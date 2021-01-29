import cv2 as cv
import numpy
import matplotlib as np
import pytesseract
from scipy.spatial import distance as dist
import os
try: 
    from PIL import Image 
except ImportError:
    import Image

#이미지 이진화
img_color = cv.imread("A2.jpg")
img_eg = cv.Canny(img_color, 100 ,200)
blur = cv.bilateralFilter(img_color,9,100,100)
img_gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 163, 255, cv.THRESH_BINARY)
img_binary = cv.bitwise_not(img_binary)

#컨투어 그림
contours,hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)

#글자 인식
filename = "{}.png".format(os.getpid()) 
cv.imwrite(filename, img_color)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Contour 영역 내에 텍스트 쓰기
# https://github.com/bsdnoobz/opencv-code/blob/master/shape-detect.cpp

max_area = 0;
ci = 0

for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    if (area > max_area):
        max_area = area
        ci = i # contour에서의 배열 번호를 차례대로 저장한다.

cnt = contours[ci] #위에서 가장 큰 영역으로 뽑힌 부분을 cnt에 저장한다.

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


color_text = label(img_color, cnt)
setLabel(img_color, color_text, cnt)

epsilon = 0.005 * cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)

size = len(approx)
print(size)

shapet = "error"

cv.line(img_color, tuple(approx[0][0]), tuple(approx[size-1][0]), (0, 255, 0), 3)
for k in range(size-1):
    cv.line(img_color, tuple(approx[k][0]), tuple(approx[k+1][0]), (0, 255, 0), 3)

if cv.isContourConvex(approx):
    if size == 3:
        setLabel(img_color, "triangle", cnt)
        shapet = 'triangle'
    elif size == 4:
        setLabel(img_color, "rectangle", cnt)
        shapet = 'rectangle'
    elif size == 5:
        setLabel(img_color, "pentagon", cnt)
        shapet = 'pentagon'
    elif size == 6:
        setLabel(img_color, "hexagon", cnt)
        shapet = 'hexagon'
    elif size == 8:
        setLabel(img_color, "octagon", cnt)
        shapet = 'octagon'
    elif size == 10:
        setLabel(img_color, "decagon", cnt)
        shapet = 'decagon'
    else:
        setLabel(img_color, "circle", cnt)
        shapet = 'circle'
else:
    setLabel(img_color, "circle", cnt)
    shapet = 'circle'

text = pytesseract.image_to_string(Image.open(filename), lang=None) 
os.remove(filename) 

print(text)
print(color_text)
print(shapet)

rere = cv.resize(img_color,(1280,600))
cv.imshow("result",rere)
cv.imwrite('test.jpg', img_color)
cv.waitKey(0)
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:42:39 2020

@author: Amanda
"""

import cv2

img = cv2.imread('teste2-Depois.png')

# img = cv2.imread('BarragemAntes.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

ret, imgThresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

altura, largura, canais = img.shape 

# print("\nDimesões: " + str(largura) + "x" + str(altura)) 

barroMax1 = (73, 83, 90)
barroMin1 = (63, 73, 80)

barroMax2 = (50, 59, 68)
barroMin2 = (40, 49, 58)

barro = (0, 0, 255)

barro1 = cv2.inRange(img, barroMax1, barroMin1)
barro2 = cv2.inRange(img, barroMax2, barroMin2)

for i in range(0, altura):
    for j in range(0, largura):
        if barro1[i,j] == imgThresh[i,j]:
            img[i,j] = barro  
        if barro2[i,j] == imgThresh[i,j]:
            img[i,j] = barro
            
cv2.imshow("Vegetacao", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
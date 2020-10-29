# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:40:58 2020

@author: Amanda
"""

import cv2

img = cv2.imread('teste2-Depois.png')

# img = cv2.imread('BarragemAntes.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, imgThresh = cv2.threshold(gray, 115, 250, cv2.THRESH_BINARY)

altura, largura, canais = img.shape 

# print("\nDimesões: " + str(largura) + "x" + str(altura)) 

verdeMax1 = (0, 255, 0)
verdeMin1 = (0, 10, 0)

verdeMax2 = (41, 74, 60)
verdeMin2 = (26, 39, 36)

verdeMax3 = (31, 42, 34)
verdeMin3 = (21, 32, 24)

verde = (0, 255, 0)
barro = (0, 0, 255)


Vegetacao1 = cv2.inRange(img, verdeMax1, verdeMin1)
Vegetacao2 = cv2.inRange(img, verdeMax2, verdeMin2)
Vegetacao3 = cv2.inRange(img, verdeMax3, verdeMin3)


for i in range(0, altura):
    for j in range(0, largura):
        if Vegetacao1[i,j] == imgThresh[i,j]:
            img[i,j] = verde  
        if Vegetacao2[i,j] == imgThresh[i,j]:
            img[i,j] = verde
        if Vegetacao3[i,j] == imgThresh[i,j]:
            img[i,j] = verde
            
cv2.imshow("Vegetacao", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# #-----------------------------------------------------------------------------

# def mostrar_inRange(img, mask):
#     imask = mask > 0
#     sliced = np.zeros_like(img, np.uint8)
#     sliced[imask] = img[imask]
#     plt.subplot(211)
#     plt.imshow(sliced)
#     plt.subplot(212)
#     plt.imshow(img)
#     plt.show()

# img = cv2.imread('teste2-Depois.png')
# # img1 = cv2.imread('BarragemAntes.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, imgThresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# altura, largura, canais = img.shape 

# print("\nDimesões: " + str(largura) + "x" + str(altura)) 

# verdeMax = (0, 255, 0)
# verdeMin = (0, 0, 0)

# verde = [0, 255, 0]


# Vegetacao = cv2.inRange(img, verdeMax, verdeMin)


# for i in range(0, altura):
#     for j in range(0, largura):
#         if Vegetacao[i,j] == imgThresh[i,j]:
#             img[i,j] = verde
            
# cv2.imshow("Vegetacao", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:50:47 2020

@author: Amanda
"""

import numpy as np
import cv2

img = cv2.imread('teste2-Depois.png')
# cv2.imwrite("BarragemDepois.png", img) 

canalAzul, canalVerde, canalVermelho = cv2.split(img)

zeros = np.zeros(img.shape[:2], dtype = "uint8")

#-----------------------------------------------------------------------------

Banda_R = cv2.merge([zeros, zeros,canalVermelho])

cv2.imshow("Vermelho", Banda_R)

cv2.imwrite("Banda_R1_Depois.png", Banda_R) 

#-----------------------------------------------------------------------------

Banda_G = cv2.merge([zeros, canalVerde, zeros])

cv2.imshow("Verde", Banda_G)

cv2.imwrite("Banda_G1_Depois.png", Banda_G) 

#-----------------------------------------------------------------------------

Banda_B = cv2.merge([canalAzul, zeros, zeros])

cv2.imshow("Azul", Banda_B)

cv2.imwrite("Banda_B1_Depois.png", Banda_B) 

#-----------------------------------------------------------------------------

cv2.imshow('Original', img)


cv2.waitKey(0)
cv2.destroyAllWindows()
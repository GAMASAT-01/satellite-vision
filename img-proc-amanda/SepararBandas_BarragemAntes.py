# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2

img = cv2.imread('teste2-Antes.png')
# cv2.imwrite("teste2-Antes.png", img) 

canalAzul, canalVerde, canalVermelho = cv2.split(img)

zeros = np.zeros(img.shape[:2], dtype = "uint8")

#-----------------------------------------------------------------------------

Banda_R = cv2.merge([zeros, zeros,canalVermelho])

cv2.imshow("Vermelho", Banda_R)

cv2.imwrite("Banda_R1_Antes.png", Banda_R) 

#-----------------------------------------------------------------------------

Banda_G = cv2.merge([zeros, canalVerde, zeros])

cv2.imshow("Verde", Banda_G)

cv2.imwrite("Banda_G1_Antes.png", Banda_G) 

#-----------------------------------------------------------------------------

Banda_B = cv2.merge([canalAzul, zeros, zeros])

cv2.imshow("Azul", Banda_B)

cv2.imwrite("Banda_B1_Antes.png", Banda_B) 

#-----------------------------------------------------------------------------

cv2.imshow('Original', img)

#-----------------------------------------------------------------------------




cv2.waitKey(0)
cv2.destroyAllWindows()
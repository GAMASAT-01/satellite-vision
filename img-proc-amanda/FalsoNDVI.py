# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:57:40 2020

@author: Amanda
"""

# import numpy as np
import cv2

G1 = cv2.imread('Banda_G_Antes.png')
R1 = cv2.imread('Banda_R_Antes.png')
B1 = cv2.imread('Banda_B_Antes.png')

FalsoNDVI_Antes = (G1 - R1) / ((G1 + R1) - B1)

cv2.imshow("falsoNDVI1", FalsoNDVI_Antes)

#------------------------------------------------------------------------------

G2 = cv2.imread('Banda_G1_Depois.png')
R2 = cv2.imread('Banda_R1_Depois.png')
B2 = cv2.imread('Banda_B1_Depois.png')

FalsoNDVI_Depois = (G2 - R2) / ((G2 + R2) - B2)

cv2.imshow("falsoNDVI2", FalsoNDVI_Depois)

#------------------------------------------------------------------------------

G3 = cv2.imread('Banda_G1_Antes.png')
R3 = cv2.imread('Banda_R1_Antes.png')
B3 = cv2.imread('Banda_B1_Antes.png')

RED1 = ((39/100) * R3)
BLUE1 = ((61/100)* B3)

# RED1 = ((2) * R3)
# BLUE1 = ((2)* B3)

TGI_Antes = (G3 - RED1 - BLUE1)

cv2.imshow("falsoNDVI3", TGI_Antes)

#------------------------------------------------------------------------------

G4 = cv2.imread('Banda_G1_Depois.png')
R4 = cv2.imread('Banda_R1_Depois.png')
B4 = cv2.imread('Banda_B1_Depois.png')

RED2 = ((39/100) * R4)
BLUE2 = ((61/100) * B4)

# RED2 = ((2) * R4)
# BLUE2 = ((2) * B4)


TGI_Depois = (G4 - RED2 - BLUE2)

cv2.imshow("falsoNDVI4", TGI_Depois)

cv2.waitKey(0)
cv2.destroyAllWindows()


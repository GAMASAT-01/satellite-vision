# import the necessary packages
from skimage.measure import compare_ssim
import imutils
import cv2

imgnum = input("Image number:")
imgpath = "dataset/sources/deforestment/{}_{}.png"
imageAfter = imgpath.format(imgnum, 'a')
imageBefore = imgpath.format(imgnum, 'b')

# load the two input images
imageA = cv2.imread(imageAfter)
imageB = cv2.imread(imageBefore)

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
diff = cv2.bitwise_not(diff)
print("SSIM: {}".format(score))

# show the output images
cv2.imshow("Bitwise not Diff", cv2.bitwise_not(diff))
cv2.waitKey(0)

# save diff to file in folder
cv2.imwrite("dataset/sources/deforestment/{}_d.jpeg".format(imgnum), diff)

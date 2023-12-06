import math
import  pandas as pd
import time
import cv2
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
plt.style.use('dark_background')


def takeImage():
   cam = cv2.VideoCapture(0)
   cv2.namedWindow("test")
   # take image on pressing 'space'
   while True:
      ret, frame = cam.read()
      cv2.imshow("test", frame)
      if not ret:
         break
      k = cv2.waitKey(1)

      if k%256 == 32:
         # SPACE pressed
         img_name = "images/takenImage.jpg"
         cv2.imwrite(img_name, frame)
         print("{} written!".format(img_name))
         break
   cam.release()
   cv2.destroyAllWindows()

   return frame


def showImgs(imgs, n_imgs, i_imgs):
   n = sqrt(n_imgs)
   m = n
   p = 1
   if n != int(n):
      n = int(n)
      m = n + 1
   print("n, m", n, m)
   fig = plt.figure()
   for i in i_imgs:
      fig.add_subplot(int(n), int(m), p)
      plt.imshow(imgs[i], cmap='gray')
      plt.axis('off')
      p += 1

def convertToEigenFaces(img):
   # Using the PCA algorithm
   pca = PCA(svd_solver='full')
   pca.fit(img)
   print(pca.components_.shape)
   print(img.shape) # (480, 640)
   best_eigenfaces = []
   for eigenface in pca.components_[0 : 40]:
      best_eigenfaces.append(eigenface.reshape(32, -1))

   showImgs(best_eigenfaces, 40, range(40))

   plt.show()

   print(best_eigenFaces.shape)
   return best_eigenfaces


img = cv2.imread('images/img1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (640, 480), fx=0.5, fy=0.5)

# show the sample image
plt.imshow(img, cmap='gray')
plt.show()

# convert sample image to eigenfaces
oEigen = convertToEigenFaces(img)
plt.show()

tImg = takeImage()
tImg = cv2.cvtColor(tImg, cv2.COLOR_BGR2GRAY)

# show taken image
plt.imshow(tImg, cmap='gray')
plt.show()

# convert taken image to eigenfaces
tEigen = convertToEigenFaces(tImg)
plt.show()



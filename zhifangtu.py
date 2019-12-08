import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
src = cv.imread("peppers.bmp")
#cv.imshow("q",src)
h,w,ch = np.shape(src)
gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
#cv.imshow("gray",gray)

pre=0
ap=[]
hest = np.zeros([256],dtype = np.int32)
for row in range(h):
    for col in range(w):
        pv = gray[row,col]
        hest[pv] +=1
        
plt.plot(hest,color = "g")
plt.xlim([0,256])
plt.show()

'''
src = cv.imread("out2.bmp")
#cv.imshow("q",src)
h,w,ch = np.shape(src)
gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
#cv.imshow("gray",gray)

pre=0
ap=[]
hest = np.zeros([256],dtype = np.int32)
for row in range(h):
    for col in range(w):
        pv = gray[row,col]
        hest[pv] +=1
        

plt.plot(hest,color = "r")
plt.xlim([0,256])

src = cv.imread("out3.bmp")
#cv.imshow("q",src)
h,w,ch = np.shape(src)
gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
#cv.imshow("gray",gray)

pre=0
ap=[]
hest = np.zeros([256],dtype = np.int32)
for row in range(h):
    for col in range(w):
        pv = gray[row,col]
        hest[pv] +=1
        

plt.plot(hest,color = "b")
plt.xlim([0,256])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()



'''











#statistics()

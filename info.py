import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
src = cv.imread("lena512.bmp")
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
        
        if col!=0:
            ap.append(int(pre)-int(pv))
        pre=pv
print(max(hest))
ap=np.array(ap)
print(np.sum(ap== -1))
akb=[]
ske=[]
for i in range(-255,255):
    akb.append(i)
    ske.append(np.sum(ap==i))




fig=plt.figure()
ax=fig.gca()

xlim(-20,+20)
plt.plot(akb,ske,color='green')
plt.scatter(akb,ske,marker='*')





src = cv.imread("aaa.bmp")
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
        
        if col!=0:
            ap.append(int(pre)-int(pv))
        pre=pv
print(max(hest))
ap=np.array(ap)
print(np.sum(ap== -1))
akb=[]
ske=[]
for i in range(-255,255):
    akb.append(i)
    ske.append(np.sum(ap==i))


xlim(-20,+20)
plt.plot(akb,ske,color='red')
plt.scatter(akb,ske)
plt.show()



'''
plt.plot(hest,color = "r")
plt.xlim([0,256])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
'''
#statistics()

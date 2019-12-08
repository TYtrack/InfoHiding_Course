import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import random
import scipy.misc

src = cv.imread("dollar.bmp")
#cv.imshow("q",src)
h,w,ch = np.shape(src)
gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
#cv.imshow("gray",gray)

'''
    list_W:list_init每个元素在list_pai的索引位置
    list_pai:排序的混沌序列
    list_init:初始混沌序列
'''
init=0.633
list_hun=[]
len_cipher=10
for i in range(len_cipher):
    init=round(1-init*init*2,4)
    list_hun.append(init)

list_pai=sorted(list_hun)
list_W=[ list_hun.index(i)for i in list_pai]

ap=[]
hest = np.zeros([256],dtype = np.int32)
for row in range(h):
    for col in range(w):
        pv = gray[row,col]
        hest[pv] +=1
        if int(pv)<212:
            ap.append(pv-8)
        elif int(pv)==212:
            ap.append(212-random.randint(0,8))
        else:
            ap.append(pv)
ap=np.array(ap)
ap=np.reshape(ap,(512,512))
cv.imwrite("dollaraaa.bmp",ap)

plt.plot(hest,color = "r")
plt.xlim([0,256])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

#statistics()

'''
进行直方图分析
'''
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


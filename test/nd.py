'''
高斯分布生成0~10000的概率
'''


import numpy as np
import matplotlib.pyplot as plt
import math
 
 
def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)


mean, sigma= 377, 100
x= np.linspace(0, 1000,1000)
  
y= normal_distribution(x, mean, sigma)
# mean2, sigma2= 0,1
# x2= np.linspace(mean2- 6*sigma2, mean2+ 6*sigma2,100)
 
# mean3, sigma3= 5,1
# x3= np.linspace(mean3- 6*sigma3, mean3+ 6*sigma3,100)

import pdb;pdb.set_trace()
# y2= normal_distribution(x2, mean2, sigma2)
# y3= normal_distribution(x3, mean3, sigma3)
 
plt.plot(x, y,'r', label='m=3777,sig=0.5')
# plt.plot(x2, y2,'g', label='m=0,sig=1')
# plt.plot(x3, y3,'b', label='m=1,sig=1')
plt.legend()
plt.grid()
plt.savefig("/home/lk/project/DALLE24Drug/CLIP4Drug/CLIP_DRP/test/1.jpg")


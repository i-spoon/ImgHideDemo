import cv2 as cv
from PIL import Image
import pywt
import numpy as np
import matplotlib.pyplot as plt
from Utils import *

# https://blog.csdn.net/yezi_happy/article/details/52804574
def Arnold(img,N=10,Decode=False):
    ''' Arnold 图像置乱
    Args:
        img: 必须是cv格式

    Returns:

    '''
    array=None
    n=cv_channel(img)
    # print(n)
    h,w=img.shape[0],img.shape[1]

    assert w==h ,"这里为了简化问题采用的都是长宽相等的问题"

    if n==3:
        array=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:
        array=img


    # print(array[0][0].shape)
    m=np.zeros((h,w,2))
    for i in range(h):
        for j in range(w):
            m[i][j] =np.array([i,j]).T

    if Decode:
        kernal = np.array([[2, -1], [-1, 1]])
    else:
        kernal=np.array([[1,1],[1,2]])
    for t in range(N):
        for i in range(h):
            for j in range(w):
                m[i][j]=np.mod(np.dot(kernal,m[i][j]),w)
    # print(m[:5][:5])
    print("Arnold位置计算完成")
    arrayNone=np.zeros_like(array,dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            t1,t2=m[i][j]
            t1,t2=int(t1),int(t2)
            arrayNone[t1][t2]=array[i][j]
    arrayNone=arrayNone.astype(np.uint8)
    # print(arrayNone[0][0])
    # print(array[0][0])

    # return cv.cvtColor(arrayNone, cv.COLOR_RGB2BGR)
    return arrayNone

def OCRML_func(x,mu):
    return mu*x*(1-x)

def OCRML_random(iter,x,xi,mu):
    x1=[0]*10
    for i in range(iter):
        x1[0]=(1-xi[0])*OCRML_func(x[0],mu)+xi[0]*OCRML_func(x[9],mu)
        for j in range(1,10):
            x1[j] = (1 - xi[j]) * OCRML_func(x[j],mu) + xi[j] * OCRML_func(x[j-1],mu)
    return x1

def OCRML_Chaos(img,
                iter=500,
                init=[0.5]*10,
                xi=[0.9]*10,
                nth=5,
                mu=4):
    ''' 单 向 耦 合 映 射 格 点 时 空 混 沌
    Args:
        img: 图像
        iter: 从多少代开始
        init: 初值序列
        xi: 1行10列参数表，【0，1】
        nth: 选择第几个，【0，9】
        mu: [3.5699456,4]
    '''
    n,w,h=cv_channel(img),img.shape[1],img.shape[0]
    # print(n)
    assert w==h,"图像长宽尽量相同"
    x=init.copy()
    x=OCRML_random(iter,x,xi,mu)
    # print(x[0],x[1])
    num,randList=h*w*n,[]
    for i in range(num):
        x=OCRML_random(1,x,xi,mu)
        # print(x[0], x[1])
        randList.append(x[nth])
    rmax,rmin=max(randList),min(randList)
    # print(randList)
    for i in range(num):
        randList[i]=(rmax-randList[i])/(rmax-rmin)*255
    # 为了二值化图像，进行特殊处理
    for i in range(num):
        if randList[i]>127:
            randList[i]=255
        else:
            randList[i]=0

    for i in range(h):
        for j in range(w):
            if n==1:
                img[i,j]=img[i,j] ^ int(randList[i*j])
            else:
                for k in range(n):
                    img[i, j, k] = img[i, j, k] ^ int(randList[i * j * k])
    return img

def ReturnTestMark(str=r'images/watermark64.png',img=None,
                   N=18,iters=500,Decode=False):
    if not Decode:
        th1=cv.imread(str)
        gray = cv.cvtColor(th1, cv.COLOR_BGR2GRAY)
        _, th1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)  # 方法选择为THRESH_OTSU
        th1=Arnold(img=th1,N=N)
        showImg(img=th1,
                winName='Arnold',
                isSave=True)
        init=[0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        xi=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.25,0.1]
        th1=OCRML_Chaos(img=th1,
                        iter=iters,
                        init=init,
                        xi=xi)
        showImg(img=th1,
                winName='OCRML_Chaos',
                isSave=True)
        return th1
    else:
        init=[0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        xi=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.25,0.1]
        th1=OCRML_Chaos(img=img,
                        iter=iters,
                        init=init,
                        xi=xi)
        showImg(img=th1,
                winName='OCRML_Chaos',
                isSave=False)
        th1=Arnold(img=th1,N=N,Decode=Decode)

        showImg(img=th1,
                winName='Arnold',
                isSave=False)
        return th1



if __name__ == '__main__':
    th1=cv.imread('./img/watermark.png')
    gray = cv.cvtColor(th1, cv.COLOR_BGR2GRAY)
    _, th1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    # print(th1[0,0,0])
    # for i in range(20):
    th1=Arnold(img=th1,N=20)
    print(th1.shape)
    showImg(img=th1)
    # img=Arnold(img=th1,N=5,Decode=True)
    # showImg(img=img)

    init=[0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    xi=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.25,0.1]
    img=OCRML_Chaos(img=th1,
                    iter=500,
                    init=init,
                    xi=xi)
    showImg(img=img)
    # img=OCRML_Chaos(img=th1,
    #                 iter=100,
    #                 init=init,
    #                 xi=xi)
    # showImg(img=img)



    # plt.subplot(131), plt.imshow(image, "gray")
    # plt.title("source image"), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.hist(image.ravel(), 256)
    # plt.title("Histogram"), plt.xticks([]), plt.yticks([])
    # ret1, th1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    # plt.subplot(133), plt.imshow(th1, "gray")
    # plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
    # plt.show()


# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 16:09:05 2022

@author: 张德昊
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from JpegCompress import KJPEG
import math
path="images/"
# 旋转攻击
def tranHack(img,theta=90):
    img=img.copy()
    h,w=img.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2), theta, 1)
    dst=cv2.warpAffine(img, M, (w,h))
    M=cv2.getRotationMatrix2D((w/2,h/2), -theta, 1)
    dst=cv2.warpAffine(dst, M, (w,h))
    return dst

# 缩放攻击
def shapeHack(img,apha=2):
    h,w=img.shape[:2]
    new_img=cv2.resize(img, (int(w/2),int(h/2)), interpolation=cv2.INTER_AREA)
    dst=cv2.resize(new_img,(w,h),interpolation=cv2.INTER_AREA)
    return dst

# 椒盐噪声攻击
def SaltyHack(img,apha=0.005,amount=0.004):
    noisyImg=np.copy(img)
    numSalt=np.ceil(amount*img.size*apha)
    coords = [np.random.randint(0,i - 1, int(numSalt)) for i in img.shape]
    noisyImg[coords] = 255
    numPepper = np.ceil(amount * img.size * (1. - apha))
    #设置添加噪声的坐标位置
    coords = [np.random.randint(0,i - 1, int(numPepper)) for i in img.shape]
    noisyImg[coords] = 0
    return noisyImg

# 高斯噪声
def GaussHack(image, mean=0, var=0.001):
  image = np.array(image/255, dtype=float)
  noise = np.random.normal(mean, var ** 0.5, image.shape)
  out = image + noise
  if out.min() < 0:
    low_clip = -1.
  else:
    low_clip = 0.
  out = np.clip(out, low_clip, 1.0)
  out = np.uint8(out*255)
  out[out>255]=255
  out[out<0]=0
  #cv.imshow("gasuss", out)
  return out

def LowerGauss(image):
    dst = cv2.GaussianBlur(image,(3,3),1.5)
    return dst

def LowerBlur(image):
    dst=cv2.blur(image,(3,3)); 
    return dst

# jpeg压缩攻击
def JpegHack(filename):
    kjpeg = KJPEG()
    kjpeg.Compress(path+filename)
    file=filename.split('.')[0]
    img=kjpeg.Decompress(path+file+".gpj")
    return img

# 裁剪攻击
def cutHack(img,width):
    h,w=img.shape[:2]
    x=np.random.randint(0,h-width)
    y=np.random.randint(0,w-width)
    img=img.copy()
    img[x:x+width,y:y+width]=0
    return img

# 计算相应的相似度
def mse(newImg,logImg):
    newImg=newImg.copy()
    logImg=logImg.copy()
    loss=newImg-logImg
    loss.reshape(-1,)
    return "{:.3f}".format(1/loss.shape[0]*np.sum(loss*loss))
    # return 1/loss.shape[0]*np.sum(loss*loss)

# 信噪比
def SNR(newImg,logImg):
    newImg=newImg.copy()
    logImg=logImg.copy()
    newImg.reshape(-1,)
    logImg.reshape(-1,)
    return  "{:.3f}".format(10*np.log10(np.sum(newImg*newImg)/np.sum(logImg*logImg)))
    # return 10*np.log10(np.sum(newImg*newImg)/np.sum(logImg*logImg))

# 峰值信噪比
def PSNR(newImg,logImg):

    pmse=float(mse(newImg,logImg))
    if pmse==0:
        return 100
    PIXEL_MAX = np.max(logImg)
    return "{:.3f}".format(20 * math.log10(PIXEL_MAX / math.sqrt(pmse)))
    # return 20 * math.log10(PIXEL_MAX / math.sqrt(pmse))

if __name__=="__main__":
    img=cv2.imread(path+"lena.jpg")
    dst=cutHack(img,160)
    cv2.imshow("img.png",dst)
    cv2.waitKey(0)
    
    
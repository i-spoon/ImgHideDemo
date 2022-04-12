import cv2
import numpy as np
import random
import math
import pywt

#仅支持灰度图，且载体尺寸，水印尺寸固定
def encodeDWT(cover_path, mark_path, res_path="./images/result.jpg"):
    cImage = cv2.imread(cover_path, 0)
    mImage = cv2.imread(mark_path, 0)
    # cImage = cv2.resize(cImage,(300,300))
    mImage = cv2.resize(mImage,(256,256))

    #DWT on cover image
    cImage =  np.float32(cImage)   
    cImage /= 255;
    coeffC = pywt.dwt2(cImage, 'haar')
    cA, (cH, cV, cD) = coeffC
    
    mImage = np.float32(mImage)
    mImage /= 255;

    #Embedding
    coeffW = (0.4*cA + 0.1*mImage, (cH, cV, cD))
    res = pywt.idwt2(coeffW, 'haar')
    cv2.imwrite(res_path, res)
    return res, cA

#离谱
def decodeDWT(mImage,cA,res_path='./images/Logo.jpg',):
    
    coeffWM = pywt.dwt2(mImage, 'haar')
    hA, (hH, hV, hD) = coeffWM

    res = (hA-0.4*cA)/0.1     #根据coeffW反解
    res *= 255
    res = np.uint8(res)
    cv2.imwrite(res_path, res)
    return res

# DCT编辑
def encodeDCT(cover_path, mark_path, d1, d2, alfa ,res_path='./images/result.jpg'):
    cImage = cv2.imread(cover_path, 0)
    mImage = cv2.imread(mark_path, 0)
    M,m = cImage.shape #512 原图的长和宽
    N,n = mImage.shape #64 水印的长和宽
    K=int(M/N) #分割倍数
    cutImage=cut(cImage,K)
    final_image=[]
    mImage=Arnold(mImage,1,1,19)
    for i in range(len(cutImage)):
        sub=[]
        for j in range(len(cutImage[i])):
            sub_dct = cv2.dct(np.float32(cutImage[i][j]))         #进行离散余弦变换
            if mImage[i][j]==0:
                sub_dct[2:6,2:6]=sub_dct[2:6,2:6]+alfa*d1
            elif mImage[i][j]==255:
                sub_dct[2:6,2:6]=sub_dct[2:6,2:6]+alfa*d2
            sub_idct = cv2.idct(sub_dct)
            sub.append(sub_idct)
        final_image.append(sub)
        res=stitch(final_image)
    cv2.imwrite(res_path, res)
    return res

#N为水印的长
def decodeDCT(img, d1,d2,N,res_path="./images/Logo.jpg"):
    M,m = img.shape #512 原图的长和宽
    K=int(M/N) #分割倍数
    cutImage=cut(img,K)
    final_image=[]
    watermark=np.zeros((64,64),dtype='int')
    for i in range(len(cutImage)):
        sub=[]
        for j in range(len(cutImage[i])):
            sub_dct = cv2.dct(np.float32(cutImage[i][j]))[2:6,2:6]
            sub1=sub_dct.reshape(1, 16)[0]
            d11=d1.reshape(1,16)[0]
            d21=d2.reshape(1,16)[0]

            if cor(sub1,d11)>=cor(sub1,d21):
                watermark[i][j]=0
            else:
                watermark[i][j]=255
    res = DeArnold(watermark,1,1,19)
    cv2.imwrite(res_path, res)
    return res

def cut(oriImg,K):
    m,n=oriImg.shape
    width=int(m/K)
    high=int(n/K)
    subImg=[]
    for i in range(0,width):
        subline=[]
        for j in range(0,high):
            sub=oriImg[i*K:(i+1)*K,j*K:(j+1)*K]
            subline.append(sub)
        subImg.append(subline)
    return subImg

def stitch(cut_image):
    i=len(cut_image)
    j=len(cut_image[0])
    height=[]
    for x in range(i):
        tmp=[]
        for y in range(j):
            tmp.append(cut_image[x][y])
        img=np.concatenate(tmp,axis=1)
        height.append(img)
    imgTmp = np.vstack(height)
    return imgTmp

#相关系数计算
def cor(z,x):
    t=0
    t1=0
    t2=0
    for i in range(len(z)):
        t+= z[i]*x[i]
        t1 += z[i]**2
        t2 += x[i]**2
    result = t/((t1*t2)**0.5)
    return result

#Arnold置乱
def Arnold(mImage,a,b,time):   
    M,m=mImage.shape
    AN = np.zeros([M,M])
    for i in range(time):
        for y in range(M):
            for x in range(m):
                xx=(x+b*y)%M
                yy=((a*x)+(a*b+1)*y)%M
                AN[yy][xx]=mImage[y][x]                
        mImage=AN.copy()
    return mImage

def DeArnold(mImage,a,b,time):    
    M,m=mImage.shape
    AN = np.zeros([M,M])
    for i in range(time):
        for y in range (M):
            for x in range(m):
                xx=((a*b+1)*x-b*y)%M
                yy=(-a*x+y)%M
                AN[yy][xx]=mImage[y][x]
            
        mImage=AN.copy()
    return mImage

if __name__ == '__main__':
    # encode4bit = encode4bit('lenaStandard.bmp', 'watermark64.png', 'encode4bit.bmp')
    # cv2.imshow('encode', encode4bit)
    # cv2.waitKey(0)
    # decode4bit = decode4bit('encode4bit.bmp', 'decode4bit.bmp')
    # cv2.imshow('decode', decode4bit)
    # cv2.waitKey(0)

    # # 嵌入时alpha越大水印越明显，提取时alpha越大图像越暗
    # encodeFFT('lenaStandard.bmp', 'watermark64.png', 'encodeFFT.jpg', 1)
    # decodeFFT('lenaStandard.bmp', 'encodeFFT.jpg', 'decodeFFT.jpg', 1)

    # encodeFFT = encodeFFT('lenaStandard.bmp', 'watermark64.png', 'encodeFFT.bmp', 1)
    # encodeFFT = np.uint8(encodeFFT)
    # cv2.imshow('encode', encodeFFT)
    # cv2.waitKey(0)
    # decodeFFT = decodeFFT('lenaStandard.bmp', 'encodeFFT.bmp', 'decodeFFT.bmp', 1)
    # decodeFFT = np.uint8(decodeFFT)
    # cv2.imshow('decode', decodeFFT)
    # cv2.waitKey(0)

    # encodeDWT, cA = encodeDWT('lenaStandard.bmp', 'watermark64.png', 'encodeDWT.bmp')
    # cv2.imshow('encode', encodeDWT)
    # cv2.waitKey(0)
    # decodeDWT = decodeDWT('encodeDWT.bmp', 'decodeDWT.bmp', cA)
    # cv2.imshow('decode', decodeDWT)
    # cv2.waitKey(0)

    d1=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    d2=np.array([[1,1,1,1],[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1]])
    encodeDCT = encodeDCT('lenaStandard.bmp', 'watermark64.png', 'encodeDCT.bmp', d1, d2, 6)
    encodeDCT = np.uint8(encodeDCT)
    cv2.imshow('encode', encodeDCT)
    cv2.waitKey(0)
    decodeDCT = decodeDCT('encodeDCT.bmp', 'decodeDCT.bmp', d1, d2, 64)
    decodeDCT = np.uint8(decodeDCT)
    cv2.imshow('decode', decodeDCT)
    cv2.waitKey(0)

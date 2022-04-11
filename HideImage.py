import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import WaterMark as wk
import Utils as ut
from numpy import linalg as lg
import math

def haar_img(img,level):
    # img_u8 = cv2.imread("./img/lena.bmp")
    n=ut.cv_channel(img)
    isGray=True
    if n==1:
        pass
    elif n==3:
        isGray=False
    else:
        assert False,"num only use 1 and 3 but n is %d"%n
    if not isGray:
        img_f32 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        img_f32 = img.astype(np.float32)
    plt.figure('二维小波一级变换')
    coeffs = pywt.dwt2(img_f32, 'haar')
    LL, (LH, HL, HH) = coeffs
    ll=[LH, HL, HH]
    for i in range(level-1):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
        ll.insert(0,HH)
        ll.insert(0,HL)
        ll.insert(0,LH)
    ll.insert(0,LL)
    ShowDWTImg(ll)
    return ll

def ShowDWTImg(ll):
    num = int((len(ll) - 1) / 3)
    LL = ll[0]
    print(num)
    for i in range(num):
        AH = np.concatenate([LL, ll[i * 3 + 1]], axis=1)
        VD = np.concatenate([ll[i * 3 + 2], ll[i * 3 + 3]], axis=1)
        LL = np.concatenate([AH, VD], axis=0)
    plt.imshow(LL, 'gray')
    plt.title('img')
    plt.show()

def Ruler_SVD(sigma,value,isPaper=False):
    if value>127:
        value=1
    else:
        value=0
    s = int(sigma)
    if isPaper==False:
        if s%2==0:
            if value==1:
                return sigma+1
            else:
                return sigma
        else:
            if value==0:
                return sigma+1
            else:
                return sigma
    else:
        if s%2==0:
            if value==0:
                return (math.floor(sigma/20)+1)*20
            else:
                return sigma
        else:
            if value==0:
                return (math.floor(sigma/20)+1)*20
            else:
                return sigma

def Paper_Encoding2(isPaper=True):
    print('读取水印')
    watermark=wk.ReturnTestMark('./img/watermark64.png',20,500)
    print('读取载体')
    image=cv2.imread('./img/lena.bmp')
    print('灰度图操作')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float)

    imgDwtList=haar_img(image,2)
    useList = [imgDwtList[4], imgDwtList[5], imgDwtList[6]]
    # useList = imgDwtList

    if ut.cv_channel(watermark)==3:
        gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
        watermark = th1.reshape(-1)
    else:
        watermark = watermark.reshape(-1)
    watermarkShape=watermark.shape[0]
    print("水印尺寸",watermarkShape)
    print(watermark)
    for A in range(len(useList)):
        tt1,tt2=useList[A].shape
        print(useList[A].shape)
        useList[A] = useList[A].reshape(-1, 4, 4)
        print(useList[A].shape)
        for i in range(watermarkShape):
            A1=useList[A][i]
            S=np.zeros_like(A1)
            U,sigma,vt=lg.svd(A1)
            sigma[-1]=Ruler_SVD(sigma[-1],watermark[i],isPaper)
            for j in range(len(sigma)):
                S[j][j]=sigma[j]
            useList[A][i]=np.dot(np.dot(U,S),vt)
        useList[A]=useList[A].reshape(tt1,tt2)
        print(useList[A].shape)
        print("************************************")

    img=pywt.idwt2((imgDwtList[0],(imgDwtList[1],imgDwtList[2],imgDwtList[3])),'haar')
    img=pywt.idwt2((img,(useList[0],useList[1],useList[2])),'haar')
    return img

def Paper_Encoding(filePathP,filePathWater,isPaper=True):
    print('读取水印')
    watermark = wk.ReturnTestMark(filePathWater, 20, 500)
    print('读取载体')
    image = cv2.imread(filePathP)
    print('灰度图操作')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float)

    imgDwtList = haar_img(image, 2)
    useList = [imgDwtList[4], imgDwtList[5], imgDwtList[6]]
    # useList = imgDwtList

    if ut.cv_channel(watermark) == 3:
        gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
        watermark = th1.reshape(-1)
    else:
        watermark = watermark.reshape(-1)
    watermarkShape = watermark.shape[0]
    print("水印尺寸", watermarkShape)
    print(watermark)
    for A in range(len(useList)):
        tt1, tt2 = useList[A].shape
        print(useList[A].shape)
        useList[A] = useList[A].reshape(-1, 4, 4)
        print(useList[A].shape)
        for i in range(watermarkShape):
            A1 = useList[A][i]
            S = np.zeros_like(A1)
            U, sigma, vt = lg.svd(A1)
            sigma[0] = Ruler_SVD(sigma[0], watermark[i], isPaper)
            for j in range(len(sigma)):
                S[j][j] = sigma[j]
            useList[A][i] = np.dot(np.dot(U, S), vt)
        useList[A] = useList[A].reshape(tt1, tt2)
        print(useList[A].shape)
        print("************************************")

    img = pywt.idwt2((imgDwtList[0], (imgDwtList[1], imgDwtList[2], imgDwtList[3])), 'haar')
    img = pywt.idwt2((img, (useList[0], useList[1], useList[2])), 'haar')
    return img
    # print(A.shape)
    #         print("奇异值",sigma.shape)
    #         print(sigma)

def Paper_Decoding(image,p=[0.5,0.25,0.25]):

    imgDwtList = haar_img(image, 2)
    useList = [imgDwtList[4], imgDwtList[5], imgDwtList[6]]
    # useList = imgDwtList
    watermark = np.zeros(4096 ,dtype=np.float32)
    for A in range(len(useList)):
        tt1, tt2 = useList[A].shape
        print(useList[A].shape)
        useList[A] = useList[A].reshape(-1, 4, 4)
        print(useList[A].shape)
        for i in range(useList[A].shape[0]):
            A1 = useList[A][i]
            S = np.zeros_like(A1)
            U, sigma, vt = lg.svd(A1)
            for j in range(len(sigma)):
                watermark[i] += int(sigma[0])%2*p[A]*255
        print(useList[A].shape)
        print("************************************")
    for i in range(watermark.shape[0]):
        if watermark[i]>128:
            watermark[i]=255
        else:
            watermark[i] = 0
    watermark=watermark.reshape(64,64).astype(np.uint8)
    return wk.ReturnTestMark(str="",img=watermark, N=20, iters=500,Decode=True)


















# def dct_test():
#     image=cv2.imread('./img/lena.bmp')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     # ut.showImg(image)
#     print(image[0,0])
#     img=image.astype(np.float32)
#     print(img[0, 0])
#     # img = haar_img(image,2)
#     # 进行离散余弦变换
#     img_dct = cv2.dct(img)
#     # 进行log处理
#     img_dct_log = np.log(abs(img_dct))
#     # 进行离散余弦反变换
#     img_idct = cv2.idct(img_dct)
#     # 进行log离散余弦反变换
#     img_idct_log = cv2.idct(img_dct_log)
#     plt.subplot(221)
#     print(image.shape)
#     # ut.showImg(img)
#     plt.imshow(image)
#     plt.subplot(222)
#     plt.imshow(img_dct)
#     plt.subplot(223)
#     plt.imshow(img_idct)
#     plt.subplot(224)
#     plt.imshow(img_dct_log)
#     plt.title('img')
#     plt.show()

if __name__ == '__main__':
    img=Paper_Encoding(isPaper=False)
    # ut.showImg(img,isCv=False,isSave=True,winName="嵌入后（min）")
    # image = cv2.imread('./img/lena.bmp')
    # ut.showImg(image, isSave=True, winName="嵌入前")
    # wk=Paper_Decoding(img)
    #     # # print(sigma)
    # # print(U.shape)
    # # print(vt.shape)
    #
    # img=np.matmul(U * sigma[:, None, :], vt)
    # tmp=np.dot(U,S)
    # img=np.dot(tmp,vt).astype(np.uint8)
    # plt.imshow(img.reshape(128,128))
    # plt.show()


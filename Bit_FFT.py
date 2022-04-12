import cv2
import numpy as np
import random
import math

def encode4bit(cover_path, mark_path, res_path="./images/result.jpg"):
    cover = cv2.imread(cover_path)
    mark = cv2.imread(mark_path)
    cMatrix = np.array(cv2.split(cover))
    cMatrix4bHigh = np.bitwise_and(cMatrix, 240)
    mMatrix = np.array(cv2.split(mark))
    mMatrix4bHigh = np.bitwise_and(mMatrix, 240)
    mMatrix4bLow = np.right_shift(mMatrix4bHigh, 4)
    # resMatrix = np.bitwise_or(cMatrix4bHigh, mMatrix4bLow)
    resMatrix = cMatrix4bHigh

    for w in range(mMatrix4bLow.shape[1]):
        for h in range(mMatrix4bLow.shape[2]):
            for k in range(3):
                resMatrix[k][w][h] = resMatrix[k][w][h] + mMatrix4bLow[k][w][h]

    res = cv2.merge(resMatrix)
    cv2.imwrite(res_path, res)
    return res

def decode4bit(img):
    iMatrix = np.array(cv2.split(img))
    iMatrix = np.bitwise_and(iMatrix, 15)
    iMatrix = np.left_shift(iMatrix, 4)
    res = cv2.merge(iMatrix)
    return res[0:64,0:64]


#im_path载体图像路径，mark_path水印图像路径，alpha混杂强度
def encodeFFT(im_path, mark_path,res_path="./images/result.jpg",alpha=1):
    # 读取源图像和水印图像
    im = cv2.imread(im_path)/255
    mark = cv2.imread(mark_path)/255
    mark=cv2.resize(mark,(im.shape[0],im.shape[1]))
    im_height, im_width, im_channel = np.shape(im)
    mark_height, mark_width = mark.shape[0], mark.shape[1]
    # 源图像傅里叶变换 可换离散小波变换
    im_f = np.fft.fft2(im)
    im_f = np.fft.fftshift(im_f)
    # 水印图像编码
    # random
    x, y = list(range(math.floor(im_height/2))), list(range(im_width))
    random.seed(im_height+im_width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(im.shape)  # 与源图像等大小的模板，用于加上水印
    for i in range(math.floor(im_height / 2)):
        for j in range(im_width):
            if x[i] < mark_height and y[j] < mark_width:
                # 对称
                tmp[i][j] = mark[x[i]][y[j]]
                tmp[im_height-i-1][im_width-j-1] = tmp[i][j]
    # 混杂
    res_f = im_f + alpha * tmp
    # 逆变换
    res = np.fft.ifftshift(res_f)
    res = np.abs(np.fft.ifft2(res)) * 255  # 回乘255
    # 保存
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])    #改变质量数值对jpg格式影响很大

#ori_path加水印前图像路径，im_path加水印后图像路径，alpha混杂强度
def decodeFFT(img, newImg, res_path='images/logo.jpg', alpha=1):
    im=newImg.copy()/255 # 其中im表示的含有加入水印的图像路径
    ori=img.copy()/255 # ori 表示的是原本的载体图像
    im_height, im_width, im_channel = np.shape(ori)
    # 源图像与水印图像傅里叶变换
    ori_f = np.fft.fft2(ori)
    ori_f = np.fft.fftshift(ori_f)
    im_f = np.fft.fft2(im)
    im_f = np.fft.fftshift(im_f)
    mark = np.abs((im_f - ori_f) / alpha)
    res = np.zeros(ori.shape)

    # 获取随机种子
    x, y = list(range(math.floor(im_height/2))), list(range(im_width))
    random.seed(im_height+im_width)
    random.shuffle(x)
    random.shuffle(y)
    for i in range(math.floor(im_height / 2)):
        for j in range(im_width):
            res[x[i]][y[j]] = mark[i][j]*255
            res[im_height-i-1][im_width-j-1] = res[i][j]

    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    res=cv2.imread(res_path)
    return res
'''本库适用于对图像的预处理调用
本文档的主要目的是为了对图像格式转化和显示提供帮助
主要格式为opencv的MAT和PIL的Image
本文档主要处理的图片格式是png和bmp（其他的没试过）
本文档主要处理的图片编码为三通道RGB和单通道灰度图（实际为二值图）
'''
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def cv_channel(img):
    '''检查 cv 格式下的通道数量
    '''
    try:
        if img.ndim==2:
            return 1
        elif img.ndim==3:
            return 3
        else:
            assert False,"图像通道数错误，请检查"
    except:
        assert False,"输入的图像不是opencv的Mat格式"

def Img2cv(img):
    ''' Image 转换为 opencv
    因为用的少这里只处理三通道的情况
    '''
    img1 = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    return img1

def cv2Img(cv1):
    '''  cv2 转换为 Image
    '''
    if cv_channel(cv1)==1:
        image = Image.fromarray(cv.cvtColor(cv1, cv.COLOR_GRAY2BGR))
    else:
        image = Image.fromarray(cv.cvtColor(cv1, cv.COLOR_BGR2RGB))
    return image

def showImg(img,winName="win",title="",isCv=True,
            axis='on',isSave=False):
    plt.figure(winName)
    # 统一格式
    if not isCv:
        image=img
        img=Img2cv(image)
        img=img.astype(np.uint8)
    image=cv2Img(img)

    plt.subplot(121)

    plt.imshow(image)
    plt.axis(axis)
    plt.title(title)
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(122)
    plt.hist(img.ravel(), 256)
    # plt.axis(False)
    plt.title(title+'灰度直方图')
    if isSave:
        plt.savefig(winName+'.png')
    plt.show()






if __name__ == '__main__':
    # img2=cv.imread('./img/lena.jpg',-1)
    img=Image.open('./img/lena.jpg')
    showImg(img,'fig',"1")
    # print(img2.shape)
    img2=Img2cv(img)
    print(img.size)

    showImg(img,"fig2","1")
    # print(np.array(img).shape)
    # showImg(img2,'fig2',"2",True,'on',True)
    # cv.imshow("cv",img2)
    # cv.waitKey()
    img3=cv2Img(img2)
    showImg(img3,'fig3',"3",False,'on',True)
    # img3.show()





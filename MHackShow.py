# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 19:01:41 2022

@author: 张德昊
"""
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from hackImg import tranHack,shapeHack,SaltyHack,GaussHack,JpegHack,cutHack,mse,SNR,PSNR,LowerGauss,LowerBlur
from HideImage import Paper_Encoding,Paper_Decoding
from PatchWorkDCT import Qlogo,Tlogo
import cv2
class DealWindows(QWidget):
    
    def PatchImage(self):
        self.flag=2
        img=cv2.imread(self.Zjpg)
        pimg=cv2.imread(self.Ljpg)
        outImg=Qlogo(img,pimg)
        self.initImg=img.copy()
        self.HackImg=outImg.copy()
        self.NewImg=outImg.copy()
        cv2.imwrite("./images/result.jpg",img)
        jpg = QtGui.QPixmap("./images/result.jpg").scaled(self.label1.width(), self.label1.height())
        self.label1.setPixmap(jpg)
        self.lineEdit1.setText(str(mse(img,self.NewImg)))
        self.lineEdit2.setText(str(SNR(self.NewImg,img)))
        self.lineEdit3.setText(str(PSNR(img,self.NewImg)))
        
    def HideImage(self):
        self.flag=1
        img=Paper_Encoding(self.Zjpg,self.Ljpg,isPaper=False)
        self.HackImg=img.copy()
        self.NewImg=img.copy()
        cv2.imwrite("./images/result.jpg",img)
        jpg = QtGui.QPixmap("./images/result.jpg").scaled(self.label1.width(), self.label1.height())
        self.label1.setPixmap(jpg)
        # self.NewImg=cv2.imread("./images/result.jpg")
        self.gray=self.tranGrey()
        self.lineEdit1.setText(str(mse(self.gray,self.NewImg)))
        self.lineEdit2.setText(str(SNR(self.NewImg,self.gray)))
        self.lineEdit3.setText(str(PSNR(self.gray,self.NewImg)))
    def tranGrey(self):
        img = cv2.imread(self.Zjpg)
        img=img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    
    def tranHackImg(self):
        if self.NewImg is not None:

            img=tranHack(self.NewImg,45)
            self.HackImg=img.copy()
            cv2.imwrite("./images/hack.jpg",img)
            jpg = QtGui.QPixmap("./images/hack.jpg").scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(jpg)
 
            
    def shapeHackImg(self):
        if self.NewImg is not None:
            img=shapeHack(self.NewImg)
            self.HackImg=img.copy()
            cv2.imwrite("./images/hack.jpg",img)
            jpg = QtGui.QPixmap("./images/hack.jpg").scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(jpg)
    
    def saltyHackImg(self):
        if self.NewImg is not None:
            img=SaltyHack(self.NewImg)
            self.HackImg=img.copy()
            cv2.imwrite("./images/hack.jpg",img)
            jpg = QtGui.QPixmap("./images/hack.jpg").scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(jpg)
    
    def GaussHackImg(self):
        if self.NewImg is not None:
            img=GaussHack(self.NewImg)
            self.HackImg=img.copy()
            cv2.imwrite("./images/hack.jpg",img)
            jpg = QtGui.QPixmap("./images/hack.jpg").scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(jpg)
            
    def JpegHackImg(self):
        if self.NewImg is not None:
            img=JpegHack("result.jpg")
            self.HackImg=img.copy()
            cv2.imwrite("./images/hack.jpg",img)
            jpg = QtGui.QPixmap("./images/hack.jpg").scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(jpg)
    
    def CutHackImg(self):
        if self.NewImg is not None:
            img=cutHack(self.NewImg,150)
            self.HackImg=img.copy()
            cv2.imwrite("./images/hack.jpg",img)
            jpg = QtGui.QPixmap("./images/hack.jpg").scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(jpg)
            
    def BoxHackImg(self):
        if self.NewImg is not None:
            img=LowerGauss(self.NewImg)
            self.HackImg=img.copy()
            cv2.imwrite("./images/hack.jpg",img)
            jpg = QtGui.QPixmap("./images/hack.jpg").scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(jpg)
            
    def GaussHackImg(self):
        if self.NewImg is not None:
            img=LowerGauss(self.NewImg)
            self.HackImg=img.copy()
            cv2.imwrite("./images/hack.jpg",img)
            jpg = QtGui.QPixmap("./images/hack.jpg").scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(jpg)
            
    def THackImg(self):
        if self.NewImg is not None:
            if self.flag==1:
                img=Paper_Decoding(self.HackImg)
                new_img=cv2.resize(img, (400,400), interpolation=cv2.INTER_AREA)
                cv2.imshow("Logo.png",new_img)
                cv2.waitKey(0)
                cv2.imwrite("./images/Logo.jpg",img)
            elif self.flag==2:
                img=Tlogo(self.HackImg,self.initImg)
                new_img=cv2.resize(img, (400,400), interpolation=cv2.INTER_AREA)
                cv2.imshow("Logo.png",new_img)
                cv2.waitKey(0)
                cv2.imwrite("./images/Logo.jpg",img)
    
    
    def __init__(self,Zjpg,Ljpg):
        self.flag=0
        super().__init__()
        self.setWindowTitle("常见的信息隐藏方法")
        self.resize(750, 550)
        self.label1=QLabel(self)
        self.label1.setFixedSize(450, 450)
        self.label1.move(20, 50)
        self.label1.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )
        self.Zjpg=Zjpg
        self.Ljpg=Ljpg
        self.NewImg=""
        
        label2=QLabel(self)
        label2.move(480,0)
        label2.setText("隐藏算法")
        btn1=QPushButton(self)
        btn1.setText("LSB")
        btn1.setFixedSize(100,40)
        btn1.move(480,20)

        btn2=QPushButton(self)
        btn2.setText("DCT")
        btn2.setFixedSize(100,40)
        btn2.move(590,20)
        
        btn3=QPushButton(self)
        btn3.setText("DFT")
        btn3.setFixedSize(100,40)
        btn3.move(480,65)
        
        btn4=QPushButton(self)
        btn4.setText("DWT")
        btn4.setFixedSize(100,40)
        btn4.move(590,65)
        
        btn5=QPushButton(self)
        btn5.setText("新型方法")
        btn5.move(480,110)
        btn5.setFixedSize(100, 40)
        btn5.clicked.connect(self.HideImage)
        label3=QLabel(self)
        label3.move(480,155)
        
        btnh=QPushButton(self)
        btnh.setText("PatchWork")
        btnh.move(590,110)
        btnh.setFixedSize(100,40)
        btnh.clicked.connect(self.PatchImage)
        
        label3.setText("常见的算法攻击")
        
        btn6=QPushButton(self)
        btn6.setText("旋转攻击")
        btn6.setFixedSize(100,40)
        btn6.move(480,175)
        btn6.clicked.connect(self.tranHackImg)
        
        btn7=QPushButton(self)
        btn7.setText("缩放攻击")
        btn7.setFixedSize(100,40)
        btn7.move(590,175)
        btn7.clicked.connect(self.shapeHackImg)
        
        btn8=QPushButton(self)
        btn8.setText("椒盐噪声")
        btn8.setFixedSize(100,40)
        btn8.move(480,220)
        btn8.clicked.connect(self.saltyHackImg)
        
        btn9=QPushButton(self)
        btn9.setText("高斯噪声")
        btn9.setFixedSize(100,40)
        btn9.move(590,220)
        btn9.clicked.connect(self.GaussHackImg)
        
        btn10=QPushButton(self)
        btn10.setText("Jpeg压缩")
        btn10.setFixedSize(100,40)
        btn10.move(480,265)
        btn10.clicked.connect(self.JpegHackImg)
        
        btn11=QPushButton(self)
        btn11.setText("裁剪攻击")
        btn11.setFixedSize(100,40)
        btn11.move(590,265)
        btn11.clicked.connect(self.CutHackImg)
        
        btn13=QPushButton(self)
        btn13.setText("盒状滤波")
        btn13.setFixedSize(100, 40)
        btn13.move(480,310)
        btn13.clicked.connect(self.BoxHackImg)
        
        btn14=QPushButton(self)
        btn14.setText("高斯滤波")
        btn14.setFixedSize(100, 40)
        btn14.move(590,310)
        btn14.clicked.connect(self.GaussHackImg)
        
        label4=QLabel(self)
        label4.move(480,355)
        label4.setText("常见的评价标准")
        
        label5=QLabel(self)
        label5.move(480,380)
        label5.setText("MSE值:")
        
        self.lineEdit1=QLineEdit(self)
        self.lineEdit1.move(540,375)
        self.lineEdit1.setFixedSize(100, 30)
        
        label6=QLabel(self)
        label6.move(480,420)
        label6.setText("SNR值:")
        self.lineEdit2=QLineEdit(self)
        self.lineEdit2.move(540,415)
        self.lineEdit2.setFixedSize(100, 30)
        
        label7=QLabel(self)
        label7.move(480,460)
        label7.setText("PSRN值:")
        self.lineEdit3=QLineEdit(self)
        self.lineEdit3.move(540,455)
        self.lineEdit3.setFixedSize(100, 30)
        
        btn12=QPushButton(self)
        btn12.setText("提取水印")           
        btn12.setFixedSize(100,40)
        btn12.move(480,500)
        btn12.clicked.connect(self.THackImg)
        
class picture(QWidget):

    def __init__(self):
        super(picture, self).__init__()

        self.resize(840, 500)
        self.setWindowTitle("信息隐藏技术")
        self.label1 = QLabel(self)
        self.label1.setFixedSize(400, 400)
        self.label1.move(20, 80)
        self.label1.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )
        
        self.label2 = QLabel(self)
        self.label2.setFixedSize(400, 400)
        self.label2.move(430, 80)

        self.label1.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )
        self.label3=QLabel(self)
        self.label3.setText("载体图片")
        self.label3.move(180,60)
        
        self.label4=QLabel(self)
        self.label4.setText("水印图片")
        self.label4.move(580,60)
        
        btn1 = QPushButton(self)
        btn1.setText("导入载体图片")
        btn1.move(20, 80)
        btn1.setFixedSize(400,400)
        btn1.clicked.connect(self.openimage1)
        
        btn2=QPushButton(self)
        btn2.setText("导入水印图片")
        btn2.move(430,80)
        btn2.setFixedSize(400,400)
        btn2.clicked.connect(self.openimage2)
        
        btn3=QPushButton(self)
        btn3.setText("初始化")
        btn3.setFixedSize(100,40)
        btn3.move(20,15)
        btn3.clicked.connect(self.btnInit)
        
        
        btn4=QPushButton(self)
        btn4.setText("载入水印")
        btn4.setFixedSize(100, 40)
        btn4.move(140,15)
        btn4.clicked.connect(self.openNewDiag)
        self.Ljpg=""
        self.Zjpg=""
        
    def openNewDiag(self):
        self.sonDiag=DealWindows(self.Zjpg,self.Ljpg)
        self.sonDiag.show()

    def openimage1(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*);;*.jpg;;*.png")
        if len(imgName)!=0:
            jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(jpg)
            self.label1.raise_()
            self.Zjpg=imgName


    def openimage2(self):
        
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*);;*.jpg;;*.png")
        if len(imgName)!=0:
            jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label1.height())
            self.label2.setPixmap(jpg)
            self.label2.raise_()
            self.Ljpg=imgName

            
    def btnInit(self):
        
        self.label1.clear()
        self.label2.clear()
        self.label1.lower()
        self.label2.lower()
        
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
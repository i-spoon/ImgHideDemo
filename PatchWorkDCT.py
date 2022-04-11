# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:55:28 2022

@author: 86177
"""
import numpy as np
import cv2
k=0.162

def arnold(img,a,b,n):
    h,w = img.shape
    N=h;
    arnoldImg = np.zeros((h,w));
    for i in range(n):
        for y in range(h):
            for x in range(w):
                xx=(x+b*y)%N
                yy=(a*x+(a*b+1)*y)%N
                arnoldImg[yy,xx]=img[y,x]
        img=arnoldImg.copy()
    return arnoldImg
        
def rearnold(arnoldImg,a,b,n):
    
    h,w = arnoldImg.shape
    img = np.zeros((h,w))
    N = h
    for i in range(n):
        for y in range(h):
            for x in range(w):
                xx=((a*b+1)*x-b*y+N)%N
                yy=(-a*x+y+N)%N
                img[yy,xx]=arnoldImg[y,x]
        arnoldImg=img.copy()
    return img

def blockProcDCT(img):
    m,n=img.shape
    img=img.copy()
    hdata=np.vsplit(img,n/8)
    for i in range(0,n//8):
        blockData=np.hsplit(hdata[i],m/8)
        for j in range(0,m//8):
            block=blockData[j]
            Yb=cv2.dct(block.astype(np.float32))
            blockData[j]=Yb.copy()
        hdata[i]=np.hstack(blockData)
    img=np.vstack(hdata)
    return img

def blockProcIDCT(img):
    m,n=img.shape
    hdata=np.vsplit(img,n/8)
    for i in range(0,n//8):
        blockData=np.hsplit(hdata[i],m/8)
        for j in range(0,m//8):
            block=blockData[j]
            Yb=cv2.idct(block.astype(np.float32))
            Yb[Yb>255]=255
            Yb[Yb<0]=0
            blockData[j]=Yb.copy()
        hdata[i]=np.hstack(blockData)
    img=np.vstack(hdata)
    return img

def Qlogo(img,w):
    r,c,t=w.shape
    IR=img[:,:,2].copy()
    IG=img[:,:,1].copy()
    IB=img[:,:,0].copy()
    
    WR=w[:,:,2].copy()
    WG=w[:,:,1].copy()
    WB=w[:,:,0].copy()


    WRA=arnold(WR,1,1,1)
    WGA=arnold(WG,1,1,1)
    WBA=arnold(WB,1,1,1)
    
    IRD=blockProcDCT(IR)
    IGD=blockProcDCT(IG)
    IBD=blockProcDCT(IB)
    
    IRDE=IRD.copy()
    IGDE=IGD.copy()
    IBDE=IBD.copy()
    
    for i in range(r):
        for j in range(c):
            x=i*8
            y=j*8;
            IRDE[x,y]=IRD[x,y]+k*WRA[i,j]
            IBDE[x,y]=IBD[x,y]+k*WBA[i,j]
            IGDE[x,y]=IGD[x,y]-k*WGA[i,j]

    IR2=blockProcIDCT(IRDE)
    IG2=blockProcIDCT(IGDE)
    IB2=blockProcIDCT(IBDE)
    print(np.max(IR2))
    I_embed = np.zeros(img.shape)
    I_embed[:,:,2] = IR2.copy();
    I_embed[:,:,1] = IG2.copy();
    I_embed[:,:,0] = IB2.copy();
    
    return I_embed.astype(np.uint8)

def Tlogo(P,img,r=64,c=64):
    

    
    PR=P[:,:,2]
    PG=P[:,:,1]
    PB=P[:,:,0]
    
    PRD=blockProcDCT(PR)
    PGD=blockProcDCT(PG)
    PBD=blockProcDCT(PB)
    WR2=np.zeros((r,c))
    WB2=np.zeros((r,c))
    WG2=np.zeros((r,c))
    
    IR=img[:,:,2].copy()
    IG=img[:,:,1].copy()
    IB=img[:,:,0].copy()
    
    IRD=blockProcDCT(IR)
    IGD=blockProcDCT(IG)
    IBD=blockProcDCT(IB)
    
    for i in range(r):
        for j in range(c):

            x=i*8
            y=j*8
            WR2[i,j]=min(255,(PRD[x,y]-IRD[x,y])/k)
            WB2[i,j]=min(255,(PBD[x,y]-IBD[x,y])/k)
            WG2[i,j]=min(255,(IGD[x,y]-PGD[x,y])/k)

    WR2=rearnold(WR2,1,1,1)
    WG2=rearnold(WG2,1,1,1)
    WB2=rearnold(WB2,1,1,1)
    
    
    W2=np.zeros((r,c,3))
    
    W2[:,:,2]=WR2.copy()
    W2[:,:,1]=WG2.copy()
    W2[:,:,0]=WB2.copy()
    return W2.astype(np.uint8)


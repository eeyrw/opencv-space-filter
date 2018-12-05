import cv2
import numpy as np

def addSaltAndPepper(src,percetage):
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.random_integers(0,src.shape[0]-1)
        randY=np.random.random_integers(0,src.shape[1]-1)
        if np.random.random_integers(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255          
    return NoiseImg 

def mediumFilter(src,kernelSize):
    filteredImage=src.copy()  
    height = src.shape[0]
    weight = src.shape[1]
    for i in range(1,height-1):
        for j in range(1,weight-1):
            kernelPixels=[]
            for kernel_i in range(-kernelSize//2,kernelSize//2):
                for kernel_j in range(-kernelSize//2,kernelSize//2):
                    kernelPixels.append(src[kernel_i+i,kernel_j+j])
            filteredImage[i,j]=np.median(kernelPixels)
    return filteredImage

def averageFilter(src,kernelSize):
    filteredImage=src.copy()  
    height = src.shape[0]
    weight = src.shape[1]
    for i in range(1,height-1):
        for j in range(1,weight-1):
            kernelPixels=[]
            for kernel_i in range(-kernelSize//2,kernelSize//2):
                for kernel_j in range(-kernelSize//2,kernelSize//2):
                    kernelPixels.append(src[kernel_i+i,kernel_j+j])
            filteredImage[i,j]=np.mean(kernelPixels)
    return filteredImage


rawImage=cv2.imread('banana.jpg',flags=cv2.IMREAD_GRAYSCALE)
cv2.imshow('Raw Image',rawImage)
noiseImage=addSaltAndPepper(rawImage,0.2)
cv2.imshow('Noise Image',noiseImage)
filteredImage=mediumFilter(noiseImage,5)
cv2.imshow('Medium Filtered Image',filteredImage)
filteredImageAvg=averageFilter(noiseImage,5)
cv2.imshow('Average Filtered Image',filteredImageAvg)
#dst = cv2.GaussianBlur(src=img, ksize=(35, 35), sigmaX=0)


cv2.waitKey(0)
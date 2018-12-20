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

def addGaussNoise(src,percetage):
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.random_integers(0,src.shape[0]-1)
        randY=np.random.random_integers(0,src.shape[1]-1)
        NoiseImg[randX,randY]=np.random.random_integers(0,255)        
    return NoiseImg 

def mediumFilter(src,kernelSize):
    filteredImage=src.copy()  
    windowRadius=kernelSize//2
    height = src.shape[0]
    weight = src.shape[1]
    for i in range(windowRadius,height-windowRadius):
        for j in range(windowRadius,weight-windowRadius):
            filteredImage[i,j]=np.median(src[i-windowRadius:i+windowRadius+1][:,j-windowRadius:j+windowRadius+1])
    return filteredImage

def averageFilter(src,kernelSize):
    filteredImage=src.copy()  
    windowRadius=kernelSize//2
    height = src.shape[0]
    weight = src.shape[1]
    for i in range(windowRadius,height-windowRadius):
        for j in range(windowRadius,weight-windowRadius):
            filteredImage[i,j]=np.mean(src[i-windowRadius:i+windowRadius+1][:,j-windowRadius:j+windowRadius+1])
    return filteredImage

def sharpenFilter(src):
    kernel=[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]] 
    kernelSize=3
    filteredImage=src.copy()  
    height = src.shape[0]
    weight = src.shape[1]
    for i in range(1,height-1):
        for j in range(1,weight-1):
            pixelValue=0
            for kernel_i in range(-kernelSize//2,kernelSize//2):
                for kernel_j in range(-kernelSize//2,kernelSize//2):
                    pixelValue=pixelValue+src[kernel_i+i,kernel_j+j]*(kernel[kernel_i+kernelSize//2][kernel_j+kernelSize//2])
            if pixelValue<0:
                pixelValue=0
            if pixelValue>255:
                pixelValue=255
            filteredImage[i,j]=pixelValue
    return filteredImage


rawImage=cv2.imread('maple.jpg',flags=cv2.IMREAD_GRAYSCALE)
cv2.imshow('Raw Image',rawImage)
noiseImage=addSaltAndPepper(rawImage,0.5)
cv2.imshow('Salt and Pepper Noise Image',noiseImage)
gaussNoiseImage=addGaussNoise(rawImage,0.5)
cv2.imshow('Gauss Noise Image',gaussNoiseImage)
filteredImage=mediumFilter(noiseImage,3)
cv2.imshow('Medium Filtered S&P Noise Image',filteredImage)
filteredImageAvg=averageFilter(noiseImage,3)
cv2.imshow('Average Filtered S&P Noise Image',filteredImageAvg)
filteredGaussNoiseImage=mediumFilter(gaussNoiseImage,3)
cv2.imshow('Medium Filtered Gauss Noise Image',filteredGaussNoiseImage)
filteredGaussNoiseImageAvg=averageFilter(gaussNoiseImage,3)
cv2.imshow('Average Filtered Gauss&P Noise Image',filteredGaussNoiseImageAvg)
filteredSharpenImage=sharpenFilter(rawImage)
cv2.imshow('Sharpen Image',filteredSharpenImage)
cv2.waitKey(0)
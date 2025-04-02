import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def histogramEqual():
    root = os.getcwd()
    img_dir = r'rep/images/9_whorl_739.jpg'
    imgPath = os.path.join(root, img_dir)
    
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    
    hist = cv.calcHist([img],[0],None,[256], [0,256])
    cdf = hist.cumsum()
    cdfNorm = cdf * float(hist.max()) / cdf.max()
    plt.figure(figsize=(20, 8)) 
    plt.subplot(231)
    plt.imshow(img, cmap = "gray")
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm, color = "b")
    plt.xlabel("p intensity")
    plt.ylabel("# of pixels")
    
    equImg = cv.equalizeHist(img)
    equhist = cv.calcHist([equImg],[0],None,[256], [0,256])
    equcdf = equhist.cumsum()
    equcdfNorm = equcdf * float(equhist.max()) / equcdf.max()
    plt.subplot(232)
    plt.imshow(equImg, cmap = "gray")
    plt.subplot(235)
    plt.plot(equhist)
    plt.plot(equcdfNorm, color = "b")
    plt.xlabel("p intensity")
    plt.ylabel("# of pixels")
    
       
    claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))
    claheImg = claheObj.apply(img)
    clahehist = cv.calcHist([claheImg],[0],None,[256], [0,256])
    clahecdf = clahehist.cumsum()
    clahecdfNorm = clahecdf * float(clahehist.max()) / clahecdf.max()
    print(type(claheImg))
    plt.subplot(233)
    plt.imshow(claheImg, cmap = "gray")
    plt.subplot(236)
    plt.plot(equhist)
    plt.plot(clahecdfNorm, color = "b")
    plt.xlabel("p intensity")
    plt.ylabel("# of pixels")
    plt.savefig("test_clahe")
    print(f"Plot saved")
def claheimg(path):
    root = os.getcwd()
    img_dir = path
    imgPath = os.path.join(root, img_dir)
    
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))
    claheImg = claheObj.apply(img)
    img_normalized = claheImg.astype(np.float32) / 255.0

    return img_normalized


    

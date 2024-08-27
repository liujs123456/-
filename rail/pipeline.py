import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
from .railutils import *

def thresholding(img):
    #setting all sorts of thresholds
    x_thresh = abs_sobel_thresh(img, orient='x', thresh_min=90 ,thresh_max=280)
    mag_thresho = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 170))
    dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = hls_select(img, thresh=(160, 255))
    lab_thresh = lab_select(img, thresh=(155, 210))
    luv_thresh = luv_select(img, thresh=(225, 255))

    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresho == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1

    # plt.imshow(threshholded, cmap='gray')  # 使用 'gray' 颜色映射以显示黑白图像
    # plt.axis('off')  # 可选：关闭坐标轴显示，使图像更整洁
    # plt.show()
    
    return threshholded

"""
    该函数主要实现了对图像进行预处理，包括阈值化、透视变换、直线检测和绘制车道区域。

    1. 将undist图像转换为numpy数组格式，并对其进行阈值化处理，得到二值化图像thresholded。
    2. 对thresholded图像进行透视变换，变换矩阵为M，变换后的图像大小与原图相同，保存为thresholded_wraped。
    3. 进行直线检测。将检测到的左右线的拟合参数更新，并保存。
    4. 在原图上绘制检测到的直线和相关信息，并将结果转换为PIL格式的图像，保存为变量area_img。

"""
def processing(img,M,Minv,left_line,right_line):
    
    #get the thresholded binary image
    img = np.array(img)
    thresholded = thresholding(img)
    #perform perspective  transform
    
    thresholded_wraped = cv2.warpPerspective(thresholded, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    left_fit, right_fit, _, _ = find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)
    # #draw the detected laneline and the information
    undist = Image.fromarray(img)
    _, _,area_img  = draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)
    
    return area_img

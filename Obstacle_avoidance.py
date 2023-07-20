import cv2
import numpy as np


def draw_line(image, focus):
    #image = np.array(image)
    # RGBtoBGR满足opencv显示格式
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    focus = 100
    actual_distance = 1
    actual_high = 1
    pixel = int(focus * actual_high / actual_distance)
    # 焦距focus =（pixel_high * actual_distance）/ actual_high
    hight = image.shape[0]
    width = image.shape[1]
    
   # print(pixel)
    
    startPointL = (pixel, pixel)
    endPointL = (pixel, hight - pixel)
    startPointR = (width - pixel, pixel)
    endPointR = (width - pixel, hight - pixel)
    startPointT = (pixel, pixel)
    endPointT = (width - pixel, pixel)
    startPointB = (pixel, hight - pixel)
    endPointB = (width - pixel, hight - pixel)
    
    # 定义线条的颜色（红色）和厚度
    color = (0, 0, 255)
    thickness = 2

    # 在帧上绘制线条
    return cv2.line(image, startPointL, endPointL, color, thickness), cv2.line(image, startPointR, endPointR, color, thickness), cv2.line(image, startPointT, endPointT, color, thickness), cv2.line(image, startPointB, endPointB, color, thickness)
    # return cv2.line(image, startPointL, endPointL, color, thickness)
    
    
    
def avoidance(top, left, bottom, right, image, focus, distance):
    image = np.array(image)
    # RGBtoBGR满足opencv显示格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    focus = 100
    actual_distance = 1
    actual_high = 1
    pixel = int(focus * actual_high / actual_distance)
    # 焦距focus =（pixel_high * actual_distance）/ actual_high
    hight = image.shape[0]
    width = image.shape[1]    
    

    cy = (top+bottom)/2
    cx = (left+right)/2
    
    count = 0
    flag = False
    place = []
    
    if cx > pixel and cx < (hight - pixel) and cy > pixel and cy < (width - pixel):
        if top < pixel:
            count += 1
        if bottom > (hight - pixel):
            count += 1
        if left < (pixel):
            count += 1
        if right > (width - pixel):
            count += 1
        if count >= 2:
            flag = True
            place.append("ahead")
        if count < 2 and distance <= 1:
            flag = True
            place.append("ahead")
    
    if cx < pixel:
        if top < pixel:
            count += 1
        if bottom > (hight - pixel):
            count += 1
        if right > (pixel):
            count += 1
        if count >= 2:
            flag = True
            place.append("left")
        if count < 2 and distance <= 1:
            flag = True
            place.append("left")
            
    if cx > (width - pixel):
        if top < pixel:
            count += 1
        if bottom > (hight - pixel):
            count += 1
        if left < (width - pixel):
            count += 1
        if count >= 2:
            flag = True
            place.append("right")
        if count < 2 and distance <= 1:
            flag = True
            place.append("right")
            
    if cy < (pixel):
        if right > (width - pixel):
            count += 1
        if bottom > (pixel):
            count += 1
        if left < (pixel):
            count += 1
        if count >= 2:
            flag = True
            place.append("top")
        if count < 2 and distance <= 1:
            flag = True
            place.append("top")

    if cy > (hight - pixel):
        if top > (hight - pixel):
            count += 1
        if right > (width - pixel):
            count += 1
        if left < (pixel):
            count += 1
        if count >= 2:
            flag = True
            place.append("bottom")
        if count < 2 and distance <= 1:
            flag = True
            place.append("bottom")

    return flag, place    



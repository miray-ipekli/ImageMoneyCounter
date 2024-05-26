# Aslıgül Kaya 1200505017
# Okan Keskin 1200505044
# Miray İpekli 1200505070

import cv2
import numpy as np

def circle_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 150)
    
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=24, param1=50, param2=30, minRadius=21, maxRadius=50)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    
    return image

def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]
    
    return image

def calculate_coins(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 150)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=50, param2=30, minRadius=21, maxRadius=50)

    coin_count = 0
    total_tl = 0
    total_euro = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            coin_count += 1
            roi = image[y - r:y + r, x - r:x + r]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contour_roi, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_roi = max(contour_roi, key=cv2.contourArea)
            contour_area = cv2.contourArea(contour_roi)

            if 6532 <= contour_area <= 7240:
                total_tl += 1
                cv2.putText(image, "1 TL", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif contour_area==5617 or contour_area==5259 or contour_area==2400:
                total_tl += 0.25
                cv2.putText(image, "25 Kurus", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif contour_area==5025 or contour_area==5527 or contour_area==5195.5 or contour_area==5501 or contour_area==5214.5 or contour_area==5530 or contour_area==5184 or contour_area==2572 or contour_area==2485.5:
                total_euro += 1
                cv2.putText(image, "1 Euro", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif contour_area==5256 or contour_area==5067.5 or contour_area==5363.5 or contour_area==4978.5 or contour_area==5547 or contour_area==2704.5 or contour_area==5003 or contour_area==4744 or contour_area==4440 or contour_area==5038 or contour_area==5256 or contour_area==5363 or contour_area==5134:
                total_euro += 0.20
                cv2.putText(image, "20 cent", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif 2581<=contour_area<=3160 or contour_area==3696.5 or contour_area==2585 or contour_area==1341 or contour_area==3627 or contour_area==3562.5 or contour_area==3148.5 or contour_area==3521 or contour_area==3288.5 or contour_area==3177 or contour_area==3836.5 or contour_area==3631.6 or contour_area==3737.5 :
                total_tl += 0.10
                cv2.putText(image, "10 Kurus", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif 1340<= contour_area <=2580 or contour_area==3319.5 or contour_area==2182.5 or contour_area==3186 or contour_area==3001.5 or contour_area==3634.5 or contour_area==3220 :
                total_tl += 0.05
                cv2.putText(image, "5 Kurus", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            

            

    return coin_count, total_tl, total_euro

def draw_coin_info(image, coin_count, total_tl, total_euro):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Coins: {coin_count}"
    cv2.putText(image, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    text = f"Total TL: {total_tl}"
    cv2.putText(image, text, (10, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    text = f"Total Euro: {total_euro}"
    cv2.putText(image, text, (10, 90), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return image

image1 = cv2.imread('1.jpg')
image2 = cv2.imread('2.jpg')
image3 = cv2.imread('3.jpg')

result1 = circle_detection(image1)
result2 = circle_detection(image2)
result3 = circle_detection(image3)

result1 = watershed_segmentation(result1)
result2 = watershed_segmentation(result2)
result3 = watershed_segmentation(result3)

cv2.imwrite('result1.jpg', result1)
cv2.imwrite('result2.jpg', result2)
cv2.imwrite('result3.jpg', result3)

coin_count1, total_tl1, total_euro1 = calculate_coins(result1)
result4 = draw_coin_info(result1, coin_count1, total_tl1, total_euro1)

coin_count2, total_tl2, total_euro2 = calculate_coins(result2)
result5 = draw_coin_info(result2, coin_count2, total_tl2, total_euro2)

coin_count3, total_tl3, total_euro3 = calculate_coins(result3)
result6 = draw_coin_info(result3, coin_count3, total_tl3, total_euro3)


cv2.imshow('Result 4', result4)
cv2.imshow('Result 5', result5)
cv2.imshow('Result 6', result6)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('result4.jpg', result4)
cv2.imwrite('result5.jpg', result5)
cv2.imwrite('result6.jpg', result6)
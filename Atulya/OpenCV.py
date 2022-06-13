import cv2
import cv2.aruco as aruco

import imutils

import numpy as np
import math


def arucoID(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    key = getattr(aruco, f"DICT_5X5_250")
    arucoDict = aruco.Dictionary_get(key)
    
    p = aruco.DetectorParameters_create()
    (c, i, r) = cv2.aruco.detectMarkers(image, arucoDict, parameters=p)
    
    return (c, i, r)


def cords(image):
    (c, i, r) = arucoID(image)
    
    if len(c) > 0:
        i = i.flatten()
        
        for (markercorner, markerid) in zip(c, i):
            corner = markercorner.reshape((4,2))
            (topleft, topright, bottomright, bottomleft) = corner
            
            topleft = (int(topleft[0]), int(topleft[1]))
            topright = (int(topright[0]), int(topright[1]))
            bottomleft = (int(bottomleft[0]), int(bottomleft[1]))
            bottomright = (int(bottomright[0]), int(bottomright[1]))
            
        return topleft, topright, bottomright, bottomleft


def angle(image):
    topleft, topright, bottomright, bottomleft = cords(image)
    
    cx = int((topleft[0]+bottomright[0])/2)
    cy = int((topleft[1]+bottomright[1])/2)
    px = int((topright[0]+bottomright[0])/2)
    py = int((topright[1]+bottomright[1])/2)
    
    m = (py-cy)/(px-cx)
    theta = math.atan(m)
    center = (cx, cy)
    
    return center, (theta*180)/math.pi


def rotate_image(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 0.8)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    
    return result


def extreme_cords(cord):
    min_c = []
    max_c = []

    for i in range(2):
        l = []
        for c in cord:
            l.append(c[i])
            
        min_c.append(min(l))
        max_c.append(max(l))

    return min_c[0], min_c[1], max_c[0], max_c[1]


def crop(image):
    topleft, topright, bottomright, bottomleft = cords(image)
    l = [topleft, topright, bottomright, bottomleft]
    
    x_min, y_min, x_max, y_max = extreme_cords(l)
    z = image[y_min:y_max, x_min:x_max]
    
    return z


img = cv2.imread('images\\CVtask.jpg')
resized_img = imutils.resize(img, width=1000)

gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

images = ['Ha', 'HaHa', 'LMAO', 'XD']
ID = {}
for image in images:
    z = cv2.imread('images\\' + image + '.jpg')
    (c, i, r) = arucoID(z)
    ID[i.tolist()[0][0]] = image

colors = {'Green':[79,209,146], 'Orange':[9,127,240], 'Black':[0,0,0], 'White':[210,222,228]}
colors_name = list(colors.keys())

colors_id = {'Green':1, 'Orange':2, 'Black':3, 'White':4}

aruco_color = {}
for i in range(1, 5):
    aruco_color[colors_name[i-1]] = ID[i]

rgb = np.ones((resized_img.shape[1],resized_img.shape[0],3))*255
new_img = cv2.resize(rgb, (resized_img.shape[1],resized_img.shape[0]))


c = []
flag = 0
for con in contours:
    perimeter = cv2.arcLength(con, True)
    approx = cv2.approxPolyDP(con, 0.01 * perimeter, True)

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = float(h)/w

        if ar >= 0.95 and ar <= 1.05:
            coordinates = [c[0].tolist() for c in approx]
            center_cord_x = (coordinates[0][0] + coordinates[2][0])//2
            center_cord_y = (coordinates[0][1] + coordinates[2][1])//2

            color_rgb = resized_img[center_cord_y, center_cord_x, :].tolist()
            color_name = colors_name[list(colors.values()).index(color_rgb)]

            x_min, y_min, x_max, y_max = extreme_cords(coordinates)
            square_img = resized_img[y_min:y_max, x_min:x_max]
            
            shape = square_img.shape
            sx = shape[0]
            sy = shape[1]
            padded_shape = (sy-30,sx-30)
            
            mid = ((coordinates[0][0]+coordinates[1][0])//2, (coordinates[0][1]+coordinates[1][1])//2)
            center = (center_cord_x, center_cord_y)
            
            try:
                slope = (mid[1]-center[1])/(mid[0]-center[0])
                theta = math.atan(slope)
            except ZeroDivisionError:
                theta = - math.pi/2

            ar = cv2.imread('images\\' + aruco_color[color_name] + '.jpg')
            center1, theta1 = angle(ar)
                
            r = rotate_image(ar, theta1-(theta*180/math.pi), center1)
            c_r = crop(r)
                
            s = cv2.resize(c_r, padded_shape)
            new_img[(y_min+15):(y_max-15), (x_min+15):(x_max-15), :] = s

            c.append(con)

            flag += 1


for con in contours[1:]:
    perimeter = cv2.arcLength(con, True)
    approx = cv2.approxPolyDP(con, 0.01 * perimeter, True)
    
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = float(h)/w

    else:
        ar = 0
        
    if ar<=0.95 or ar >= 1.05:
        coordinates = [c[0].tolist() for c in approx]
        x_min, y_min, x_max, y_max = extreme_cords(coordinates)
        
        if len(approx) > 6:
            s = resized_img[y_min:y_max, (x_min-10):(x_max+10)]
            new_img[y_min:y_max, (x_min-10):(x_max+10), :] = s
        else:
            s = resized_img[y_min:y_max, (x_min):(x_max)]
            new_img[y_min:y_max, x_min:x_max, :] = s


cv2.drawContours(new_img, c, -1, (255,0,0), 3)
cv2.imwrite('Final.jpg', new_img) 

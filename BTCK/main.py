# Dựa trên phương pháp trong ppsd.png
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
import os 
from lib import *


resolution = 60 #2.5cm ~ 60pixel
dpath = './raw'
img_case = 13
# print(len(os.listdir(dpath)))

rawBGR = cv2.imread(dpath+'/'+str(img_case)+'.jpg')
rawBGR = rawBGR[400:1050,100:750]
rawRGB = cv2.cvtColor(rawBGR,cv2.COLOR_BGR2RGB)

# rawRGB = np.array(rawRGB,dtype=np.float64)/255
hsv = cv2.cvtColor(rawRGB,cv2.COLOR_RGB2HSV)

# Ngưỡng tách bề mặt bên trên 
lowest_o = np.array([70,30,90])
highest_o = np.array([100,120,200])
mask_o = cv2.inRange(hsv,lowest_o,highest_o,cv2.THRESH_BINARY)

# plt.imshow(mask_o)
# plt.show()

k = 0 #phat hien toa do goc 
corner_x = 0
corner_y = 0 
for i in range(len(mask_o)):
    corner_y+=1
    for j in range(0,len(mask_o[i])-1):
        if mask_o[i][j]>0 and mask_o[i+1][j-1] <255 and mask_o[i+2][j-1]<255:
            if mask_o[i][j+100]>0: 
                # print (j,corner_y)
                corner_x = j
                k =1
                break
    if k==1:
        img = cv2.circle(rawRGB,(corner_x,corner_y), 5, (0,255,0), -1)
        break 

centroid_over = [corner_x+resolution,corner_y+resolution]
img = cv2.circle(rawBGR,(centroid_over[0],centroid_over[1]), 10, (0,255,0), -1)
# cv2.imwrite('./img_latex/cen_o.jpg',img)

# plt.imshow(img)
# plt.show()

#
offset_r = 0

lowest_r = np.array([0,105,105])
highest_r = np.array([9,130,135])
mask_r = cv2.inRange(hsv,lowest_r,highest_r,cv2.THRESH_BINARY)

sum_r = []
k_r = [] 
for i in range(corner_y, corner_y+2*resolution):
    sum = 0
    for j in range(corner_x+2*resolution, corner_x+3*resolution):
        if mask_r[i][j]>0 and mask_r[i][j+1]>0:
            sum+=1
    if sum > 0:
        sum_r.append(sum)

if len(sum_r) == 0:
    offset_r = 0
else:
    offset_r = max(sum_r)

# print(f'Offset right: {offset_r}')

#
offset_f = 0

lowest_f = np.array([0,150,180])
highest_f = np.array([20,200,230])
mask_f = cv2.inRange(hsv,lowest_f,highest_f,cv2.THRESH_BINARY)

offset_f = 0 
for i in range(corner_y-int(resolution/2),corner_y):
    sum = np.sum(mask_f[i])/255
    if sum > 3:
        offset_f+=1

# print(f'Offset forward: {offset_f}')

centroid_under = [centroid_over[0]+offset_r, centroid_over[1]-offset_f]
print(f'Centroid {img_case+1} in pixel: {centroid_under}')
img = cv2.circle(img,(centroid_under[0], centroid_under[1]), 10, (0,0,0), -1)

# plt.imshow(img)
# cv2.imwrite('./img_latex/cen_u.jpg',img)
# plt.show()

# 
p_matrix, img_points, r_matrix, t_vector, k_matrix, dist_coef = cameraCalib()
b = cvt2Dto3D(np.array([centroid_under[0], centroid_over[1]]),r_matrix,t_vector,k_matrix)
print(f'Centroid {img_case+1} in cm: {b}')

#
ref = []
with open('ref2.txt','r') as f:
    for line in f:
        a = line.split()
        ref.append(a)

err = [float(ref[img_case][2]) - float(b[0]), float(ref[img_case][3]) - float(b[1])]
print(f'Error in cm: {err}')


c = cvt3Dto2D(np.array([b]),p_matrix)
# print(c)








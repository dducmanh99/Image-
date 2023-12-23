from lib import * 
import matplotlib.pyplot as plt


corner=[]
with open('calib_corner.txt','r') as f:
    for line in f: 
        a = line.split()
        corner.append(a)

p_matrix, img_points, r_matrix, t_vector, k_matrix, dist_coef = cameraCalib()

err_x = []
err_y = []
for i in range(7):
    for j in range(7):
        # print (corner[i*7+j])
        err_x.append( abs(j * 2.5 - float(corner[i*7+j][0])))
        err_y.append( abs(i * 2.5 - float(corner[i*7+j][1])))
        # print(i,j)

count = list(range(0,49,1))
plt.plot(count,err_x,color='red',label='Sai số trục x')
plt.plot(count, err_y,color='blue',label='Sai số trục y')
plt.xlabel("Thứ tự góc trên bàn cờ")
plt.ylabel("Sai số (cm)")
plt.title("Sai số của các góc trên bàn cờ khi hiệu chỉnh")
plt.legend()
plt.grid()
plt.savefig("calib_err.jpg")
plt.show()


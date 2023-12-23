import cv2 
import numpy as np
import glob 

# Kích thước ô bàn cờ (mm)
square_dim = 25
# Cờ để lưu dữ liệu 
save_ref = 0 

# Hàm tìm ma trận hiệu chỉnh ảnh
def cameraCalib():
    # kích thước bàn cờ
    CHECKERBOARD = (7,7) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #
    obj_points = []
    #
    img_points = []

    #
    obj_p = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    # print (obj_p)
    obj_p[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    # print(obj_p)
    prev_img_shape = None
    # print (obj_p[0,:,:2])

    # Đọc ảnh calib 
    images = './calib/5.jpg'
    img = cv2.imread(images)
    # Sử dụng phần trung tâm của ảnh
    img = img[400:1050,100:750]
    cv2.imwrite('./calib/in.jpg',img)
    # img = cv2.resize(img,(300,400))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # ret : cờ để xác định trạng thái calib, corner: tọa độ các góc trên bàn cờ
    ret,corners = cv2.findChessboardCorners(gray,CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        obj_points.append(obj_p)
        corners2 = corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        # print(corners2)
        img_points.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    # img = cv2.circle(img,(153,161), 10, (255,0,0), -1)
    # print(f'Toa do (0,0,0) tren anh : {img_points[0][0]}')
    
    # cv2.imshow('img',img)
    # cv2.imwrite('./calib/out.jpg',img)
    # cv2.waitKey(0)

    # Các ma trận hiệu chỉnh ảnh 
    ret, k_matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    r_matrix = cv2.Rodrigues(rvecs[0])[0]
    t_vector = tvecs[0]/10
    rt_matrix = np.column_stack((r_matrix, t_vector))
    p_matrix = np.matmul(k_matrix,rt_matrix) # A[R|t]

    return p_matrix, img_points, r_matrix, t_vector, k_matrix, dist_coef
    
# Hàm chuyển tọa độ 2D pixel --> tọa độ 3D thực tế dựa trên phương trình chuyển đổi ảnh calib_matrix.png
def cvt2Dto3D (point: np.ndarray, r_matrix:np.ndarray, t_vector: np.ndarray, k_matrix: np.ndarray):
    camera_matrix_inv = np.linalg.inv(k_matrix)
    vec_1 = np.array([point[0]*t_vector[-1], point[1] * t_vector[-1], t_vector[-1]]).reshape((3, 1))
    camera_point = np.dot(camera_matrix_inv, vec_1)
    t_vector = t_vector.reshape((3, 1))
    vec_2 = camera_point - t_vector
    rotation_matrix_inv = np.linalg.inv(r_matrix)
    world_point = (np.dot(rotation_matrix_inv, vec_2) * square_dim).reshape(-1)
    world_point[-1] = 0.0
    return world_point

# Hàm chuyển đổi tọa độ 3D thực tế --> tọa độ 2D pixel
def cvt3Dto2D(point3D: np.ndarray, p_matrix: np.ndarray):
    point = np.append(point3D.copy()/square_dim, 1.0).reshape((4,1))
    point_not_norm = np.matmul(p_matrix,point)
    result = np.array([np.floor(point_not_norm[0]/point_not_norm[-1]), np.floor(point_not_norm[1]/point_not_norm[-1])])

    return result.astype(np.int16).reshape(-1)

# Kiểm tra các hàm bên trên 
p_matrix, img_points, r_matrix, t_vector, k_matrix, dist_coef = cameraCalib()

# b = cvt2Dto3D(np.array([497.6342,497.44913]),r_matrix,t_vector,k_matrix) #384 499
# print(f'Toa do thuc te: {b}')
# print(b[0])

# c = cvt3Dto2D(np.array([0,0,0]),p_matrix)

# for i in img_points[0]:

# print(img_points[0][0][0][0])

# Lưu các tọa độ góc vào .txt 
if save_ref == 1:
    for i in img_points[0]:
        # print(i[0][1])
        b = cvt2Dto3D(np.array([i[0][0],i[0][1]]),r_matrix,t_vector,k_matrix) #384 499
        print(i[0][0],i[0][1])
        print(b)
        with open('a.txt','a') as f:
            f.write(str(b[0])+' '+str(b[1])+' '+str(i[0][0])+' '+str(i[0][1])+'\n')

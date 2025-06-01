# 기존 코드
#!/usr/bin/env python3
import rclpy
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from yolo_msgs.msg import BoundingBoxArray, BoundingBox
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import math
import numpy as np

def Rz(p):
    return np.array([
        [np.cos(p), -np.sin(p), 0],
        [np.sin(p), np.cos(p), 0],
        [0, 0, 1]
    ])

def Rx(t):
    return np.array([
        [1, 0, 0],
        [0, np.cos(t), -np.sin(t)],
        [0, np.sin(t), np.cos(t)]
    ])

def Ry(r):
    return np.array([
        [np.cos(r), 0, np.sin(r)],
        [0, 1, 0],
        [-np.sin(r), 0, np.cos(r)]
    ])

class Republisher(Node):
    def __init__(self):
        super().__init__('inference_republisher')
        self.cv_bridge = CvBridge()
        self.img = None # 이미지 받을 곳
        self.lidar_data = np.zeros((221, 2)) # 라이다 받을 곳
        self.box = []
        self.angle = []
        self.distance = []
        self.filter = LaserScan()


        self.u, self.v = None, None
        self.img_sub = self.create_subscription(Image, '/inference/image_raw', self.img_callback, 10)
        self.imgdata_sub = self.create_subscription(BoundingBoxArray, '/inference/object_raw', self.imgdata_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan,'/scan', self.lidar_callback,10)
        self.publisher_point = self.create_publisher(Float64MultiArray, '/point', 10)

        self.send_scan = self.create_publisher(LaserScan, '/scan_filter', 10)
        self.timer = self.create_timer(1, self.timer_callback)

        self.inc_angle = 0.25
        self.camera = np.array([[-0.03],[0.0],[0.07]]) # 라이다 좌표계 기준 카메라 위치
        
        # 팬(pan) 각도와 틸트(tilt) 각도 (라디안 단위)
        p = np.radians(90)  # Z축 90 (팬)
        t = np.radians(-90)   # X축 -90 (틸트)
        roll = np.radians(0)  # Y축 0 (롤)

        self.R = Rz(p) @ Rx(t)
        self.R_inv = np.linalg.inv(self.R)
        self.msg = Float64MultiArray()

        # 카메라 파라미터
        self.K = np.array([[804.78, 0, 311.65], [0, 806.04, 220.08], [0, 0, 1]])

    def img_callback(self, msg):
        self.img = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

    def imgdata_callback(self, msg):
        self.box.clear()
        for bounding_box in msg.bounding_box_array:
            self.box.append((bounding_box.x_max, bounding_box.x_min, bounding_box.y_max,bounding_box.y_min))

    def lidar_callback(self, msg):
        if self.img is not None :
            self.filter = msg
            self.angle.clear()
            self.distance.clear()

            for i in range (221):
                if len(self.box) > 0:
                    project_lidar_to_image(msg.ranges[610 + i], 152.5  + self.inc_angle * i, self)
                    if self.u is not None and self.v is not None:
                        catch(self, msg.ranges[610 + i], i)
                elif len(self.box) == 0:
                    self.filter.ranges[610 + i] = 1000.0
            self.send_scan.publish(self.filter)
            data = Float64MultiArray()
            
            data.layout.dim.append(MultiArrayDimension(label = "rows", size = 2, stride = len(self.angle) + len(self.distance)))
            data.layout.dim.append(MultiArrayDimension(label = "cols", size = len(self.angle), stride = 1))
            data.data = self.angle + self.distance

            self.publisher_point.publish(data)
            Fusion(self.img)

    def timer_callback(self):
        self.box.clear()

def project_lidar_to_image(distance, angle, self):
    # 1. 라이다 데이터를 월드 좌표계로 변환
    world_point = lidar_to_world(distance, angle) # 맞음

    # 2. 월드 좌표계를 카메라 좌표계로 변환
    camera_point = self.R_inv @ (world_point - self.camera)

    # 3. 카메라 좌표계를 이미지 평면으로 투영
    image_point = project_to_image(camera_point, self)
    u, v = int(image_point[0]), int(image_point[1])

    # 이미지 범위 내인지 확인하고 점을 그리기
    if 0 <= u < self.img.shape[1] and 0 <= v < self.img.shape[0]:
        self.u, self.v = u, v
        cv2.circle(self.img, (u, v), 3, (0, 0, 255), -1)

def lidar_to_world(distance, angle):
    # 각도를 라디안으로 변환
    theta = np.deg2rad(angle)
    x = distance * np.cos(theta)
    y = distance * np.sin(theta)
    z = 0  # 2D 라이다 데이터는 z = 0
    return np.array([[x], [y], [z]])

def project_to_image(camera_point, self):
    # 카메라 내재 파라미터 행렬을 사용하여 투영
    image_point_homogeneous = self.K @ camera_point
    # 동차 좌표계를 2D 좌표로 변환
    image_point = image_point_homogeneous[:2] / image_point_homogeneous[2]
    return image_point

def catch(self, dis, count):
    state = 0
    for i in range(len(self.box)):
        x_max, x_min, y_max, y_min = self.box[i]
        if x_max >= self.u >= x_min and y_max >= self.v >= y_min and state == 0 :
            cv2.circle(self.img, (self.u, self.v), 3, (255, 0, 0), -1)
            self.filter.ranges[610 + count] = dis
            ang = (610 + count)/4
            self.angle.append(ang)
            self.distance.append(dis)
            
            min_ang = pixel_to_lidar_angle(x_min)*4
            max_ang = pixel_to_lidar_angle(x_max)*4
            min_dis = min([r for r in self.filter.ranges[max_ang:min_ang:] if r > 0])

            if dis == min_dis :
                cv2.line(self.img, (311, 2700), (self.u, self.v), (0,255,0), 2)
                cv2.putText(self.img, f"{min_dis*100:.2f}cm", (self.u, self.v-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            break
        else :
            self.filter.ranges[610 + count] = 1000.0

def pixel_to_lidar_angle(x, image_width=640, fov=55):
    center = image_width // 2
    angle = 180 - ((x - center) / (image_width / 2)) * (fov / 2)
    return int(angle)

def Fusion(image):
    cv2.imshow("Fusion", image)
    if cv2.waitKey(1) == ord('q'):
        raise StopIteration


def main(args=None):
    rclpy.init(args=args)

    republisher = Republisher()
   
    rclpy.spin(republisher)
    
    republisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
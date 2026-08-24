# ROS2 SensorFusion and Time Sync

LiDAR 및 카메라 센서 데이터의 동기화(Time Synchronization)와 센서 퓨전(Sensor Fusion)을 수행하는 ROS 2 기반 프로젝트입니다.

---

## 📦 주요 패키지 구성 (Packages)

1. **`sensorfusion_pkg`** (Python)
   - 카메라/YOLO 검출 메시지와 LiDAR 데이터를 결합하여 센서 융합 처리를 수행합니다.
   - `republisher`, `sensorfusion1`, `sensorfusion2` 노드 포함

2. **`time_sync_cpp`** (C++)
   - `message_filters`를 이용해 이기종 센서 토픽 간 타임스탬프를 정밀하게 동기화하는 C++ 노드(`time_sync_node`)

3. **`time_sync_py`** (Python)
   - Python 환경에서 센서 토픽 간 타임 동기화를 처리하는 노드(`time_sync_node`)

---

## ⚙️ 개발 환경 (Environment)

- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble
- **Python**: 3.10+
- **주요 의존성**: `rclcpp`, `rclpy`, `sensor_msgs`, `std_msgs`, `message_filters`, `cv_bridge`, `yolo_msgs`

---

## 🛠️ 빌드 방법 (Build)

```bash
# 워크스페이스 빌드
cd ~/ros2_ws
colcon build --symlink-install

# 환경 설정 로드
source install/setup.bash
```

---

## 🚀 실행 가이드 (Usage)

### 1. 타임 동기화 (Time Synchronization)
```bash
# C++ 동기화 노드 실행
ros2 run time_sync_cpp time_sync_node

# Python 동기화 노드 실행
ros2 run time_sync_py time_sync_node
```

### 2. 센서 퓨전 (Sensor Fusion)
```bash
# 센서 퓨전 노드 실행
ros2 run sensorfusion_pkg sensorfusion1
# 또는
ros2 run sensorfusion_pkg sensorfusion2
```

## 실행 영상
[![2D Lidar - Camera Sensor Fusion (Time Sync)](https://youtu.be/98R6KRzKUC4/0.jpg)](https://youtu.be/98R6KRzKUC4)

## 참고 및 출처

- yolov5 패키지의 일부 코드는 아래 프로젝트를 참고하거나 수정하였습니다.  
  [wannn-one/yolov5-ros2 (GitHub)](https://github.com/wannn-one/yolov5-ros2)
  - 라이선스: Apache License 2.0 (LICENSE 파일 포함)

- lakibeam1 패키지의 일부 코드는 아래 프로젝트를 참고하거나 수정하였습니다.  
  [RichbeamTechnology/Lakibeam_ROS2_Driver (GitHub)](https://github.com/RichbeamTechnology/Lakibeam_ROS2_Driver)
  - 라이선스: MIT License (LICENSE 파일 포함)

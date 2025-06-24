#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import message_filters
from sensor_msgs.msg import Temperature, FluidPressure

class TimeSyncNode(Node):
    def __init__(self):
        super().__init__('time_sync_node')
        
        # QoS 설정
        qos = QoSProfile(depth=10)
        
        # Publisher 생성
        self.pub_temp = self.create_publisher(Temperature, 'temp', qos)
        self.pub_pres = self.create_publisher(FluidPressure, 'pressure', qos)
        
        # message_filters Subscriber 생성
        self.sub_temp = message_filters.Subscriber(self, Temperature, 'temp', qos_profile=qos)
        self.sub_pres = message_filters.Subscriber(self, FluidPressure, 'pressure', qos_profile=qos)
        
        # ApproximateTimeSynchronizer 생성 (큐 크기 10, 슬롭 50ms)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_temp, self.sub_pres], 
            queue_size=10, 
            slop=0.05  # 50ms 슬롭 설정
        )
        self.sync.registerCallback(self.on_sync)
        
        # Timer 생성
        self.timer1 = self.create_timer(0.5, self.publish_temperature)  # 500ms
        self.timer2 = self.create_timer(0.1, self.publish_fluid_pressure)  # 100ms
        
        # 동기화 통계 변수
        self.sync_count = 0
        self.accumulated_diff_ns = 0
    
    def on_sync(self, t, p):
        """동기화 콜백 함수"""
        # 두 메시지 헤더 스탬프 차이 계산 (절대값, ns)
        t_ns = int(t.header.stamp.nanosec)
        p_ns = int(p.header.stamp.nanosec)
        diff_ns = abs(t_ns - p_ns)
        
        self.accumulated_diff_ns += diff_ns
        self.sync_count += 1
        
        avg_diff_ms = (self.accumulated_diff_ns / self.sync_count) * 1e-6
        
        self.get_logger().info(
            f'동기화 성공 [{self.sync_count}회]: '
            f'temp={t.header.stamp.sec}.{t.header.stamp.nanosec}, '
            f'fluid={p.header.stamp.sec}.{p.header.stamp.nanosec}, '
            f'시간차={diff_ns * 1e-6:.3f}ms, 평균={avg_diff_ms:.3f}ms'
        )
    
    def publish_temperature(self):
        """Temperature 메시지 발행"""
        now = self.get_clock().now()
        self.get_logger().info(f'publish_Temperature Time: {now.nanoseconds} ns')
        
        msg = Temperature()
        msg.header.stamp = now.to_msg()
        msg.temperature = 25.0
        
        self.pub_temp.publish(msg)
    
    def publish_fluid_pressure(self):
        """FluidPressure 메시지 발행"""
        now = self.get_clock().now()
        self.get_logger().info(f'publish_FluidPressure Time: {now.nanoseconds} ns')
        
        msg = FluidPressure()
        msg.header.stamp = now.to_msg()
        msg.fluid_pressure = 101325.0
        
        self.pub_pres.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TimeSyncNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

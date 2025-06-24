#include "rclcpp/rclcpp.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "sensor_msgs/msg/temperature.hpp"
#include "sensor_msgs/msg/fluid_pressure.hpp"
using namespace std::chrono_literals;

class TimeSyncNode : public rclcpp::Node
{
public:
    TimeSyncNode() : Node("time_sync_node")
    {
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
        pub_temp = create_publisher<sensor_msgs::msg::Temperature>("temp", qos);
        pub_pres = create_publisher<sensor_msgs::msg::FluidPressure>("pressure", qos);
        sub_temp.subscribe(this, "temp", qos.get_rmw_qos_profile());
        sub_pres.subscribe(this, "pressure", qos.get_rmw_qos_profile());

        // ApproximateTimeSynchronizer 생성 (큐 크기 10, 슬롭 50ms)
        sync = std::make_shared<message_filters::Synchronizer<
            message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::Temperature, sensor_msgs::msg::FluidPressure>>>(
            message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::Temperature, sensor_msgs::msg::FluidPressure>(10),
            sub_temp, sub_pres);
        sync->setAgePenalty(0.00005); // 0.05 = 50ms 슬롭 설정

        sync->registerCallback(
            std::bind(&TimeSyncNode::on_sync, this, std::placeholders::_1, std::placeholders::_2));

        timer1 = create_wall_timer(500ms, std::bind(&TimeSyncNode::publish_Temperature, this));
        timer2 = create_wall_timer(100ms, std::bind(&TimeSyncNode::publish_FluidPressure, this));
    }

private:
    // 동기화 통계 변수
    size_t sync_count = 0;
    int64_t accumulated_diff_ns = 0;

    void on_sync(
        const sensor_msgs::msg::Temperature::ConstSharedPtr &t,
        const sensor_msgs::msg::FluidPressure::ConstSharedPtr &p)
    {
        // 1) 두 메시지 헤더 스탬프 차이 계산 (절대값, ns)
        int64_t t_ns = static_cast<int64_t>(t->header.stamp.nanosec);
        int64_t p_ns = static_cast<int64_t>(p->header.stamp.nanosec);
        int64_t diff_ns = std::llabs(t_ns - p_ns);

        accumulated_diff_ns += diff_ns;
        ++sync_count;

        double avg_diff_ms = (accumulated_diff_ns /
                              static_cast<double>(sync_count)) *
                             1e-9;

        RCLCPP_INFO(this->get_logger(),
                    "동기화 성공 [%zu회]: temp=%u.%u, fluid=%u.%u, 시간차=%.3fms, 평균=%.3fms",
                    sync_count,
                    t->header.stamp.sec, t->header.stamp.nanosec,
                    p->header.stamp.sec, p->header.stamp.nanosec,
                    diff_ns * 1e-6, avg_diff_ms);
    }

    void publish_Temperature()
    {
        auto now = get_clock()->now();
        RCLCPP_INFO(this->get_logger(),
                    "publish_Temperature Time: %llu ns",
                    static_cast<uint64_t>(now.nanoseconds()));
        sensor_msgs::msg::Temperature msg;
        msg.header.stamp = now;
        msg.temperature = 25.0;
        pub_temp->publish(msg);
    }

    void publish_FluidPressure()
    {
        auto now = get_clock()->now();
        RCLCPP_INFO(this->get_logger(),
                    "publish_FluidPressure Time: %llu ns",
                    static_cast<uint64_t>(now.nanoseconds()));
        sensor_msgs::msg::FluidPressure msg;
        msg.header.stamp = now;
        msg.fluid_pressure = 101325.0;
        pub_pres->publish(msg);
    }

    message_filters::Subscriber<sensor_msgs::msg::Temperature> sub_temp;
    message_filters::Subscriber<sensor_msgs::msg::FluidPressure> sub_pres;
    std::shared_ptr<
        message_filters::Synchronizer<
            message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::Temperature, sensor_msgs::msg::FluidPressure>>>
        sync;
    rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr pub_temp;
    rclcpp::Publisher<sensor_msgs::msg::FluidPressure>::SharedPtr pub_pres;
    rclcpp::TimerBase::SharedPtr timer1, timer2;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TimeSyncNode>());
    rclcpp::shutdown();
    return 0;
}

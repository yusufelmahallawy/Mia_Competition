#!/usr/bin/env python3
"""
Converts the single-beam LaserScan published by the Gazebo ultrasonic
sensor emulation (/ultrasonic_scan) into an std_msgs/Int32 message
on /ultrasonic_distance (distance in centimeters), matching the
message type your autonomous_motion_node subscribes with.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32


class UltrasonicRangeConverter(Node):
    def __init__(self):
        super().__init__('ultrasonic_range_converter')
        self.sub = self.create_subscription(
            LaserScan, '/ultrasonic_scan', self.scan_callback, 10)
        self.pub = self.create_publisher(Int32, '/ultrasonic_distance', 10)

    def scan_callback(self, msg: LaserScan):
        if len(msg.ranges) > 0 and not math.isnan(msg.ranges[0]) and not math.isinf(msg.ranges[0]):
            distance_m = msg.ranges[0]
        else:
            distance_m = msg.range_max  # nothing detected -> report max range

        distance_cm = int(round(distance_m * 100.0))

        out = Int32()
        out.data = distance_cm
        self.pub.publish(out)


def main():
    rclpy.init()
    node = UltrasonicRangeConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

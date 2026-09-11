#!/usr/bin/env python3

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ManualPilotNode(Node):

    def __init__(self):
        super().__init__("manual_pilot_node")

        # Publish to a separate topic.
        # Integration Node will forward this to /cmd_vel
        # when MANUAL mode is active.
        self.cmd_pub = self.create_publisher(
            Twist,
            "/manual_cmd_vel",
            10
        )

        self.linear_speed = 0.20
        self.angular_speed = 0.60

        self.get_logger().info(
            "Manual Pilot Node started"
        )
        self.get_logger().info(
            "Publishing on /manual_cmd_vel"
        )

    def publish_twist(
        self,
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0
    ):
        msg = Twist()

        msg.linear.x = linear_x
        msg.linear.y = linear_y
        msg.angular.z = angular_z

        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish_twist(0.0, 0.0, 0.0)

    def get_key(self):
        tty.setraw(sys.stdin.fileno())

        key = sys.stdin.read(1)

        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            self.settings
        )

        return key

    def run(self):

        self.settings = termios.tcgetattr(sys.stdin)

        try:
            while rclpy.ok():

                if select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.05
                )[0]:

                    key = self.get_key()

                    # Forward
                    if key.lower() == "w":
                        self.publish_twist(
                            linear_x=self.linear_speed
                        )

                    # Backward
                    elif key.lower() == "s":
                        self.publish_twist(
                            linear_x=-self.linear_speed
                        )

                    # Strafe left
                    elif key.lower() == "a":
                        self.publish_twist(
                            linear_y=self.linear_speed
                        )

                    # Strafe right
                    elif key.lower() == "d":
                        self.publish_twist(
                            linear_y=-self.linear_speed
                        )

                    # Rotate left
                    elif key.lower() == "q":
                        self.publish_twist(
                            angular_z=self.angular_speed
                        )

                    # Rotate right
                    elif key.lower() == "e":
                        self.publish_twist(
                            angular_z=-self.angular_speed
                        )

                    # Stop
                    elif key == " ":
                        self.stop()

                    # Exit
                    elif key.lower() == "x":
                        self.stop()
                        break

                rclpy.spin_once(
                    self,
                    timeout_sec=0.01
                )

        finally:
            self.stop()

            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                self.settings
            )

    def destroy_node(self):
        self.stop()
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = ManualPilotNode()

    try:
        node.run()

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
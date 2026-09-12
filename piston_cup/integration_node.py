#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger

from pynput import keyboard


class IntegrationNode(Node):

    def __init__(self):
        super().__init__('integration_node')

        # ==========================================
        # CURRENT MODE
        # ==========================================
        self.mode = 'MANUAL'

        # ==========================================
        # PUBLISHER TO ROBOT
        # ==========================================
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # ==========================================
        # MANUAL COMMANDS
        # ==========================================
        self.manual_sub = self.create_subscription(
            Twist,
            '/manual_cmd_vel',
            self.manual_callback,
            10
        )

        # ==========================================
        # AUTONOMOUS COMMANDS
        # ==========================================
        self.auto_sub = self.create_subscription(
            Twist,
            '/auto_cmd_vel',
            self.auto_callback,
            10
        )

        # ==========================================
        # AUTONOMOUS START SERVICE
        # ==========================================
        self.start_auto_client = self.create_client(
            Trigger,
            '/start_autonomous'
        )

        # ==========================================
        # KEYBOARD
        # ==========================================
        self.listener = keyboard.Listener(
            on_press=self.on_press
        )

        self.listener.start()

        # ==========================================
        # STARTUP
        # ==========================================
        self.get_logger().info(
            '========================================'
        )
        self.get_logger().info(
            'Integration Node started'
        )
        self.get_logger().info(
            'Current Mode: MANUAL'
        )
        self.get_logger().info(
            'M = Manual'
        )
        self.get_logger().info(
            'N = Autonomous'
        )
        self.get_logger().info(
            'R = Reset'
        )
        self.get_logger().info(
            '========================================'
        )

    # ==========================================
    # MANUAL CALLBACK
    # ==========================================
    def manual_callback(self, msg):

        if self.mode == 'MANUAL':
            self.cmd_pub.publish(msg)

    # ==========================================
    # AUTONOMOUS CALLBACK
    # ==========================================
    def auto_callback(self, msg):

        if self.mode == 'AUTO':
            self.cmd_pub.publish(msg)

    # ==========================================
    # KEYBOARD CONTROL
    # ==========================================
    def on_press(self, key):

        try:

            # ==================================
            # MANUAL MODE
            # ==================================
            if key.char == 'm':

                self.mode = 'MANUAL'

                self.get_logger().info(
                    'MODE → MANUAL'
                )

            # ==================================
            # AUTONOMOUS MODE
            # ==================================
            elif key.char == 'n':

                self.mode = 'AUTO'

                self.get_logger().info(
                    'MODE → AUTO'
                )

                self.start_autonomous()

            # ==================================
            # RESET
            # ==================================
            elif key.char == 'r':

                self.reset_robot()

        except AttributeError:
            pass

    # ==========================================
    # START AUTONOMOUS
    # ==========================================
    def start_autonomous(self):

        if not self.start_auto_client.wait_for_service(
                timeout_sec=1.0):

            self.get_logger().error(
                '/start_autonomous service not available'
            )

            return

        request = Trigger.Request()

        future = self.start_auto_client.call_async(request)

        future.add_done_callback(
            self.autonomous_start_response
        )

    # ==========================================
    # AUTONOMOUS START RESPONSE
    # ==========================================
    def autonomous_start_response(self, future):

        try:

            response = future.result()

            if response.success:

                self.get_logger().info(
                    'Autonomous Node started successfully'
                )

            else:

                self.get_logger().warn(
                    f'Autonomous start failed: '
                    f'{response.message}'
                )

        except Exception as e:

            self.get_logger().error(
                f'Failed to start autonomous: {e}'
            )

    # ==========================================
    # RESET
    # ==========================================
    def reset_robot(self):

        # Stop command
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)

        self.get_logger().warn(
            'RESET'
        )

        self.get_logger().warn(
            'Return the robot physically to the start position.'
        )

        self.get_logger().warn(
            f'Current mode remains: {self.mode}'
        )

    # ==========================================
    # MAIN
    # ==========================================
def main(args=None):

    rclpy.init(args=args)

    node = IntegrationNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        # Stop robot before shutdown
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)

        node.listener.stop()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
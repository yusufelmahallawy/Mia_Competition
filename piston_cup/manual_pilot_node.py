import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pynput import keyboard


class ManualPilotNode(Node):

    def __init__(self):
        super().__init__('manual_pilot_node')

        # Integration Node will receive this
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

        print("Manual Pilot started!")
        print("W: Forward")
        print("S: Backward")
        print("A: Strafe Left")
        print("D: Strafe Right")
        print("Ctrl+C: Exit")

    def on_press(self, key):
        msg = Twist()

        try:
            if key.char == 'w':
                msg.linear.x = 0.5
            elif key.char == 's':
                msg.linear.x = -0.5
            elif key.char == 'a':
                msg.linear.y = 0.5
            elif key.char == 'd':
                msg.linear.y = -0.5
            else:
                return

            self.cmd_pub.publish(msg)

        except AttributeError:
            pass

    def on_release(self, key):
        msg = Twist()

        try:
            if key.char in ['w', 's', 'a', 'd']:
                print("--STOP")
                self.cmd_pub.publish(msg)
        except AttributeError:
            pass    

def main(args=None):

    rclpy.init(args=args)

    node = ManualPilotNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        # Stop robot before shutdown
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)

        node.listener.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




    #source /opt/ros/jazzy/setup.bash
    #ros2 run ros_gz_bridge parameter_bridge \
    #/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist

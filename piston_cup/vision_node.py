import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO

from ament_index_python.packages import get_package_share_directory

import os


class VisionNode(Node):

    def __init__(self):
        model_path = os.path.join(
            get_package_share_directory('piston_cup'),
            'models',
            'best.pt'
        )
        self.model = YOLO(model_path)
        super().__init__('vision_node')

        # Convert ROS Image <-> OpenCV image
        self.bridge = CvBridge()

        # Subscribe to camera images
        self.subscription = self.create_subscription(
            Image,
            '/mono/image',
            self.image_callback,
            10
        )

        self.get_logger().info('Vision node started')

    def image_callback(self, msg):

        # Convert ROS image to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.get_logger().info('Received an image frame')

        # Run YOLO detection
        results = self.model(frame, verbose=False)

        # Process detections
        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = self.model.names[class_id]

                self.get_logger().info(
                    f'{class_name} | '
                    f'confidence: {confidence:.2f} | '
                    f'bbox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})'
                )


def main(args=None):

    rclpy.init(args=args)

    node = VisionNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import os
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge
from ultralytics import YOLO

from ament_index_python.packages import get_package_share_directory


# Detections below this confidence are ignored
CONFIDENCE_THRESHOLD = 0.5


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # ------------------------------------------------------------
        # Load YOLO model
        # ------------------------------------------------------------
        self.model = YOLO('/home/piston_cup_ws/src/piston_cup/models/best.pt')

        # ------------------------------------------------------------
        # ROS Image <-> OpenCV
        # ------------------------------------------------------------
        self.bridge = CvBridge()

        # ------------------------------------------------------------
        # Subscribe to camera
        # ------------------------------------------------------------
        self.subscription = self.create_subscription(
            Image,
            '/mono/image',
            self.image_callback,
            10
        )

        # ------------------------------------------------------------
        # Publish best scroll detection
        #
        # Data:
        # [found, x1, y1, x2, y2, confidence]
        #
        # found = 1.0 -> detection found
        # found = 0.0 -> no detection
        # ------------------------------------------------------------
        self.detection_pub = self.create_publisher(
            Float32MultiArray,
            '/scroll_detection',
            10
        )

        self.get_logger().info('Vision node started')

    def image_callback(self, msg):

        # ------------------------------------------------------------
        # Camera image -> grayscale
        # The YOLO model was trained on grayscale images.
        # ------------------------------------------------------------
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='mono8'
        )

        # ------------------------------------------------------------
        # YOLO detection
        # ------------------------------------------------------------
        results = self.model(
            frame,
            verbose=False
        )

        # Best detection, whether it is REAL or FAKE
        best_box = None
        best_conf = 0.0
        best_class_name = None

        # ------------------------------------------------------------
        # Find highest-confidence detection
        # ------------------------------------------------------------
        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = self.model.names[class_id]

                # Ignore low-confidence detections
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Keep the highest-confidence detection
                if confidence > best_conf:

                    best_conf = confidence
                    best_class_name = class_name
                    best_box = box.xyxy[0].tolist()

        # ------------------------------------------------------------
        # Prepare detection message
        # ------------------------------------------------------------
        out = Float32MultiArray()

        if best_box is not None:

            x1, y1, x2, y2 = best_box

            # Publish:
            # [found, x1, y1, x2, y2, confidence]
            out.data = [
                1.0,
                x1,
                y1,
                x2,
                y2,
                best_conf
            ]
            self.get_logger().info(
                f'{best_class_name} found | '
                f'confidence: {best_conf:.2f} | '
                f'bbox: '
                f'({x1:.0f}, {y1:.0f}, '
                f'{x2:.0f}, {y2:.0f})'
            )

            # --------------------------------------------------------
            # Draw bounding box
            # --------------------------------------------------------
            x1, y1, x2, y2 = map(
                int,
                [x1, y1, x2, y2]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                255,
                2
            )

            # Show class + confidence on screen
            label = f'{best_class_name} {best_conf:.2f}'

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                255,
                2
            )

        else:

            # No detection
            out.data = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ]

        # ------------------------------------------------------------
        # Publish detection
        # ------------------------------------------------------------
        self.detection_pub.publish(out)

        # ------------------------------------------------------------
        # Display camera image
        # ------------------------------------------------------------
        cv2.imshow(
            'Robot Camera - YOLO',
            frame
        )

        cv2.waitKey(1)

    def destroy_node(self):

        cv2.destroyAllWindows()

        super().destroy_node()


def main():

    rclpy.init()

    node = VisionNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__== '__main__':
    main()
import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

CONFIDENCE_THRESHOLD = 0.5

class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # Load YOLO model
        model_path = os.path.join(get_package_share_directory('piston_cup'),'models','final_final.pt')
        self.model = YOLO(model_path)

        self.bridge = CvBridge()        # ROS Image <-> OpenCV

        self.subscription = self.create_subscription(Image,'/mono/image',self.image_callback,10)
        self.detection_pub = self.create_publisher(Float32MultiArray,'/scroll_detection',10)
        self.get_logger().info('Vision node started')

    def image_callback(self, msg):
        gray_frame = self.bridge.imgmsg_to_cv2(msg,desired_encoding='mono8')    # Get image as grayscale

        # YOLO receives the grayscale image
        results = self.model(gray_frame,verbose=False)

        # Make a BGR copy ONLY for displaying colored boxes/text
        frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)

        best_box = None
        best_conf = 0.0
        best_class_name = None

        # Search for the best detection
        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                if confidence > best_conf:

                    best_conf = confidence
                    best_class_name = class_name
                    best_box = box.xyxy[0].tolist()

        # Create output message
        out = Float32MultiArray()

        if best_box is not None:
            x1, y1, x2, y2 = best_box

            # Publish detection -> [found, x1, y1, x2, y2, confidence]
            out.data = [1.0,x1,y1,x2,y2,best_conf]

            # Print detection
            self.get_logger().info(
                f'{best_class_name} found | '
                f'confidence: {best_conf:.2f} | '
                f'bbox: 'f'({x1:.0f}, {y1:.0f}, 'f'{x2:.0f}, {y2:.0f})')

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int,[x1, y1, x2, y2])

            # Choose box/text color
            if best_class_name == 'real':
                color = (0, 0, 255)       #red

            else:
                color = (255, 0, 0)       # Fake -> Blue

            # Draw bounding box
            cv2.rectangle(frame,(x1, y1),(x2, y2),color,2)

            # Draw label
            label = f'{best_class_name} {best_conf:.2f}'

            cv2.putText(frame,label,(x1, max(y1 - 10, 20)),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        else:
            # No detection
            out.data = [0.0,0.0,0.0,0.0,0.0,0.0]

        self.detection_pub.publish(out)

        # Show camera image
        cv2.imshow('Robot Camera - YOLO',frame)

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

if __name__ == '__main__':
    main()
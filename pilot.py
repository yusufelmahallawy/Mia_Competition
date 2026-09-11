#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import termios
import tty

# إعداد مفاتيح التحكم السريع (W: قدام، S: لورا، A: يمين، D: شمال، Space: وقوف)
moveBindings = {
    'w': (1.0, 0.0),
    's': (-1.0, 0.0),
    'a': (0.0, 1.0),
    'd': (0.0, -1.0),
    ' ': (0.0, 0.0)
}

def getKey():
    # قراءة الحرف المدخل من لوحة المفاتيح مباشرة بدون الحاجة للضغط على Enter
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

class PilotInterface:
    def __init__(self):
        rospy.init_node('pilot_interface_node', anonymous=True)
        
        # تهيئة الـ Publisher لإرسال أوقات السرعة والحركة
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # تهيئة الـ Subscriber لاستقبال صور الكاميرا الأحادية
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/mono/image', Image, self.image_callback)
        
        self.linear_speed = 0.2  # سرعة التقدم
        self.angular_speed = 0.5 # سرعة الالتفاف

    def image_callback(self, data):
        try:
            # تحويل رسالة ROS لـ OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # عرض الصورة على النافذة
        cv2.imshow("Pilot View - Mono Camera", cv_image)
        cv2.waitKey(1)

    def run(self):
        print("جاري تشغيل واجهة التحكم... استخدم أزرار W, A, S,D للحركة و المسافة للوقوف. اضغط Ctrl+C للخروج.")
        x = 0.0
        th = 0.0

        try:
            while not rospy.is_shutdown():
                key = getKey()
                if key in moveBindings.keys():
                    x = moveBindings[key][0]
                    th = moveBindings[key][1]
                else:
                    if key == '\x03': # Ctrl+C للخروج
                        break
                
                # بناء رسالة الحركة Twist
                twist = Twist()
                twist.linear.x = x * self.linear_speed
                twist.linear.y = 0.0
                twist.linear.z = 0.0
                twist.angular.x = 0.0
                twist.angular.y = 0.0
                twist.angular.z = th * self.angular_speed
                
                # إرسال الأمر للروبوت
                self.cmd_pub.publish(twist)

        except Exception as e:
            print(e)

        finally:
            # إيقاف الروبوت عند الخروج
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = PilotInterface()
        node.run()
    except rospy.ROSInterruptException:
        pass
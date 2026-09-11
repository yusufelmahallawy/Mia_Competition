#!/usr/bin/env python3

"""
autonomous_motion_node.py

Piston Cup Finals
Member 2 - Autonomous Motion Control

Publishes:
    /auto_cmd_vel
    /scroll_count

Subscribes:
    /ultrasonic_distance
    /mono/image

Services:
    /start_autonomous
    /reset_autonomous

IMPORTANT:
This node does NOT publish directly to /cmd_vel.
Integration Node is responsible for forwarding /auto_cmd_vel
to /cmd_vel.
"""

import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Int32
from std_srvs.srv import Trigger

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None


# ============================================================
# CONFIG
# ============================================================

class Config:

    FORWARD_SPEED = 0.15
    APPROACH_SPEED = 0.08
    STRAFE_SPEED = 0.15
    ROTATE_SPEED = 0.6

    TURN_90_DURATION = 1.3
    ROW_SCAN_MAX_DURATION = 6.0
    RETURN_TIMEOUT_FAILSAFE = 8.0

    SAFETY_STOP_DISTANCE = 0.12
    APPROACH_STOP_DISTANCE_FAR = 0.40
    APPROACH_STOP_DISTANCE_NEAR = 0.30

    CANDIDATE_OBSTACLE_DISTANCE = 2.05
    KNOWN_WALL_DISTANCE = 1.50

    MAX_DETECTION_RETRIES = 3
    RETRY_BACKUP_DISTANCE = 0.08

    LOOP_DT = 0.05


# ============================================================
# STATES
# ============================================================

class State(Enum):

    WAITING = auto()

    CHECK_INITIAL_PATH = auto()

    # Branch A
    A_ROTATE_TO_ROW = auto()
    A_SCAN_ROW = auto()
    A_APPROACH_1 = auto()
    A_DETECT_1 = auto()
    A_RESUME_ROW = auto()
    A_APPROACH_2 = auto()
    A_DETECT_2 = auto()

    # Branch B
    B_MOVE_TO_OBSTACLE = auto()
    B_DETECT_1 = auto()
    B_ROTATE_TO_RETURN = auto()
    B_RETURN_TO_START = auto()
    B_ROTATE_TO_SEARCH2 = auto()
    B_SCAN_FOR_CANDIDATE = auto()
    B_APPROACH_2 = auto()
    B_DETECT_2 = auto()

    DONE = auto()
    STUCK = auto()


# ============================================================
# NODE
# ============================================================

class AutonomousMotionNode(Node):

    def __init__(self):

        super().__init__("autonomous_motion_node")

        # ----------------------------------------------------
        # Publishers
        # ----------------------------------------------------

        self.cmd_pub = self.create_publisher(
            Twist,
            "/auto_cmd_vel",
            10
        )

        self.scroll_count_pub = self.create_publisher(
            Int32,
            "/scroll_count",
            10
        )

        # ----------------------------------------------------
        # Subscribers
        # ----------------------------------------------------

        self.create_subscription(
            Range,
            "/ultrasonic_distance",
            self.ultrasonic_callback,
            10
        )

        self.create_subscription(
            Image,
            "/mono/image",
            self.image_callback,
            10
        )

        # ----------------------------------------------------
        # Services
        # ----------------------------------------------------

        self.start_service = self.create_service(
            Trigger,
            "/start_autonomous",
            self.start_autonomous_callback
        )

        self.reset_service = self.create_service(
            Trigger,
            "/reset_autonomous",
            self.reset_autonomous_callback
        )

        # ----------------------------------------------------
        # Variables
        # ----------------------------------------------------

        self.bridge = CvBridge() if CvBridge else None

        self.latest_range_m = None
        self.latest_cv_image = None

        self.scrolls_found = 0

        self.state = State.WAITING

        self.autonomous_enabled = False

        self.get_logger().info(
            "Autonomous Motion Node ready"
        )

    # ========================================================
    # SENSOR CALLBACKS
    # ========================================================

    def ultrasonic_callback(self, msg):

        self.latest_range_m = msg.range

    def image_callback(self, msg):

        if self.bridge is None:
            return

        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Camera conversion error: {e}"
            )

    # ========================================================
    # START / RESET
    # ========================================================

    def start_autonomous_callback(self, request, response):

        if self.autonomous_enabled:

            response.success = False
            response.message = "Autonomous already running"

            return response

        self.scrolls_found = 0
        self.publish_scroll_count()

        self.state = State.CHECK_INITIAL_PATH

        self.autonomous_enabled = True

        self.get_logger().info(
            "AUTONOMOUS STARTED"
        )

        response.success = True
        response.message = "Autonomous started"

        return response

    def reset_autonomous_callback(self, request, response):

        self.stop()

        self.autonomous_enabled = False

        self.scrolls_found = 0
        self.state = State.WAITING

        self.publish_scroll_count()

        self.get_logger().warn(
            "AUTONOMOUS RESET"
        )

        response.success = True
        response.message = "Autonomous reset"

        return response

    # ========================================================
    # ROS LOOP
    # ========================================================

    def spin_and_sleep(self, dt=None):

        rclpy.spin_once(
            self,
            timeout_sec=0.0
        )

        time.sleep(
            dt if dt is not None else Config.LOOP_DT
        )

    # ========================================================
    # COMMANDS
    # ========================================================

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

        self.publish_twist(
            0.0,
            0.0,
            0.0
        )

    # ========================================================
    # SCROLL COUNT
    # ========================================================

    def publish_scroll_count(self):

        msg = Int32()

        msg.data = self.scrolls_found

        self.scroll_count_pub.publish(msg)

    def increment_scroll_count(self):

        self.scrolls_found += 1

        self.publish_scroll_count()

        self.get_logger().info(
            f"Scroll detected: {self.scrolls_found}"
        )

    # ========================================================
    # SAFETY
    # ========================================================

    def is_safety_triggered(self):

        return (
            self.latest_range_m is not None
            and self.latest_range_m
            < Config.SAFETY_STOP_DISTANCE
        )

    # ========================================================
    # MOTION
    # ========================================================

    def move_timed(
        self,
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0,
        duration=0.0,
        stop_on_safety=True
    ):

        start = time.time()

        while (
            time.time() - start < duration
            and rclpy.ok()
            and self.autonomous_enabled
        ):

            if (
                stop_on_safety
                and self.is_safety_triggered()
            ):

                self.stop()

                self.get_logger().warn(
                    "Safety stop triggered"
                )

                return False

            self.publish_twist(
                linear_x,
                linear_y,
                angular_z
            )

            self.spin_and_sleep()

        self.stop()

        return self.autonomous_enabled

    def rotate_90(self, direction=1):

        return self.move_timed(
            angular_z=Config.ROTATE_SPEED * direction,
            duration=Config.TURN_90_DURATION
        )

    def move_forward_until_distance(
        self,
        target_distance_m,
        max_duration,
        speed=None
    ):

        if speed is None:
            speed = Config.FORWARD_SPEED

        start = time.time()

        while (
            rclpy.ok()
            and self.autonomous_enabled
        ):

            if self.is_safety_triggered():

                self.stop()

                return "safety_stop"

            if time.time() - start > max_duration:

                self.stop()

                self.get_logger().warn(
                    "Fail-safe timeout reached"
                )

                return "timeout_failsafe"

            if (
                self.latest_range_m is not None
                and self.latest_range_m <= target_distance_m
            ):

                self.stop()

                return "reached_target"

            self.publish_twist(
                linear_x=speed
            )

            self.spin_and_sleep()

        return "stopped"

    # ========================================================
    # VISION
    # ========================================================

    def detect_scroll(self):

        """
        Temporary Vision hook.

        Member 1 should replace this with the real
        detection model.

        Returns:
            found: bool
            bbox: (x, y, w, h) or None
            confidence: float
        """

        if self.latest_cv_image is None:

            return False, None, 0.0

        # TODO:
        # Connect Member 1's trained model here.

        return False, None, 0.0

    # ========================================================
    # VISUAL SERVO
    # ========================================================

    def visual_servo_approach(
        self,
        stop_distance_m
    ):

        while (
            rclpy.ok()
            and self.autonomous_enabled
        ):

            if self.is_safety_triggered():

                self.stop()

                return False

            if (
                self.latest_range_m is not None
                and self.latest_range_m <= stop_distance_m
            ):

                self.stop()

                return True

            found, bbox, _ = self.detect_scroll()

            if (
                not found
                or self.latest_cv_image is None
            ):

                self.publish_twist(
                    linear_x=Config.APPROACH_SPEED
                )

                self.spin_and_sleep()

                continue

            frame_width = self.latest_cv_image.shape[1]

            x, y, w, h = bbox

            box_center_x = x + w / 2.0
            frame_center_x = frame_width / 2.0

            offset = (
                box_center_x
                - frame_center_x
            )

            deadband_px = frame_width * 0.08

            lateral_gain = 0.15

            if abs(offset) < deadband_px:

                self.publish_twist(
                    linear_x=Config.APPROACH_SPEED
                )

            else:

                strafe = (
                    lateral_gain
                    if offset > 0
                    else -lateral_gain
                )

                self.publish_twist(
                    linear_x=Config.APPROACH_SPEED * 0.5,
                    linear_y=strafe
                )

            self.spin_and_sleep()

        return False

    # ========================================================
    # DETECTION RETRY
    # ========================================================

    def attempt_detection_with_retry(self):

        for attempt in range(
            Config.MAX_DETECTION_RETRIES
        ):

            found, _, _ = self.detect_scroll()

            if found:

                return True

            self.get_logger().info(
                f"Detection attempt {attempt + 1} failed"
            )

            self.move_timed(
                linear_x=-Config.APPROACH_SPEED,
                duration=1.0,
                stop_on_safety=True
            )

        self.get_logger().warn(
            "All detection retries exhausted"
        )

        return False

    # ========================================================
    # STATE MACHINE
    # ========================================================

    def run(self):

        while (
            rclpy.ok()
            and self.autonomous_enabled
        ):

            # ------------------------------------------------
            # INITIAL CHECK
            # ------------------------------------------------

            if self.state == State.CHECK_INITIAL_PATH:

                if self.latest_range_m is None:

                    self.spin_and_sleep()

                    continue

                path_clear = (
                    self.latest_range_m
                    > Config.CANDIDATE_OBSTACLE_DISTANCE
                )

                if path_clear:

                    self.state = State.A_ROTATE_TO_ROW

                else:

                    self.state = State.B_MOVE_TO_OBSTACLE

            # ------------------------------------------------
            # BRANCH A
            # ------------------------------------------------

            elif self.state == State.A_ROTATE_TO_ROW:

                self.rotate_90(direction=1)

                self.state = State.A_SCAN_ROW

            elif self.state == State.A_SCAN_ROW:

                found, _, _ = self.detect_scroll()

                if found:

                    self.state = State.A_APPROACH_1

                else:

                    if not self.move_timed(
                        linear_x=Config.FORWARD_SPEED,
                        duration=0.3
                    ):

                        self.state = State.STUCK

            elif self.state == State.A_APPROACH_1:

                if self.visual_servo_approach(
                    Config.APPROACH_STOP_DISTANCE_FAR
                ):

                    self.state = State.A_DETECT_1

                else:

                    self.state = State.STUCK

            elif self.state == State.A_DETECT_1:

                if self.attempt_detection_with_retry():

                    self.increment_scroll_count()

                    self.state = State.A_RESUME_ROW

                else:

                    self.state = State.A_SCAN_ROW

            elif self.state == State.A_RESUME_ROW:

                found, _, _ = self.detect_scroll()

                if found:

                    self.state = State.A_APPROACH_2

                else:

                    self.move_timed(
                        linear_x=Config.FORWARD_SPEED,
                        duration=0.3
                    )

            elif self.state == State.A_APPROACH_2:

                if self.visual_servo_approach(
                    Config.APPROACH_STOP_DISTANCE_FAR
                ):

                    self.state = State.A_DETECT_2

                else:

                    self.state = State.STUCK

            elif self.state == State.A_DETECT_2:

                if self.attempt_detection_with_retry():

                    self.increment_scroll_count()

                    self.state = State.DONE

                else:

                    self.state = State.A_RESUME_ROW

            # ------------------------------------------------
            # BRANCH B
            # ------------------------------------------------

            elif self.state == State.B_MOVE_TO_OBSTACLE:

                result = self.move_forward_until_distance(
                    Config.APPROACH_STOP_DISTANCE_FAR,
                    Config.ROW_SCAN_MAX_DURATION
                )

                if result == "timeout_failsafe":

                    self.state = State.STUCK

                else:

                    self.state = State.B_DETECT_1

            elif self.state == State.B_DETECT_1:

                if self.attempt_detection_with_retry():

                    self.increment_scroll_count()

                    self.state = State.B_ROTATE_TO_RETURN

                else:

                    self.state = State.STUCK

            elif self.state == State.B_ROTATE_TO_RETURN:

                self.rotate_90(direction=-1)

                self.state = State.B_RETURN_TO_START

            elif self.state == State.B_RETURN_TO_START:

                result = self.move_forward_until_distance(
                    Config.KNOWN_WALL_DISTANCE,
                    Config.RETURN_TIMEOUT_FAILSAFE
                )

                if result == "timeout_failsafe":

                    self.state = State.STUCK

                else:

                    self.state = State.B_ROTATE_TO_SEARCH2

            elif self.state == State.B_ROTATE_TO_SEARCH2:

                self.rotate_90(direction=1)

                self.state = State.B_SCAN_FOR_CANDIDATE

            elif self.state == State.B_SCAN_FOR_CANDIDATE:

                if (
                    self.latest_range_m is not None
                    and self.latest_range_m
                    < Config.CANDIDATE_OBSTACLE_DISTANCE
                ):

                    self.state = State.B_APPROACH_2

                else:

                    if not self.move_timed(
                        linear_x=Config.FORWARD_SPEED,
                        duration=0.3
                    ):

                        self.state = State.STUCK

            elif self.state == State.B_APPROACH_2:

                if self.visual_servo_approach(
                    Config.APPROACH_STOP_DISTANCE_NEAR
                ):

                    self.state = State.B_DETECT_2

                else:

                    self.state = State.STUCK

            elif self.state == State.B_DETECT_2:

                if self.attempt_detection_with_retry():

                    self.increment_scroll_count()

                    self.state = State.DONE

                else:

                    self.state = State.B_SCAN_FOR_CANDIDATE

            # ------------------------------------------------
            # DONE
            # ------------------------------------------------

            elif self.state == State.DONE:

                self.stop()

                self.get_logger().info(
                    "AUTONOMOUS PHASE COMPLETE"
                )

                self.get_logger().info(
                    f"Scrolls detected: {self.scrolls_found}"
                )

                self.autonomous_enabled = False

                break

            # ------------------------------------------------
            # STUCK
            # ------------------------------------------------

            elif self.state == State.STUCK:

                self.stop()

                self.get_logger().error(
                    "ROBOT STUCK - waiting for reset"
                )

                self.autonomous_enabled = False

                break

            self.spin_and_sleep()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def main_loop(self):

        while rclpy.ok():

            if not self.autonomous_enabled:

                self.stop()

                rclpy.spin_once(
                    self,
                    timeout_sec=0.1
                )

                continue

            self.run()

    # ========================================================
    # CLEANUP
    # ========================================================

    def destroy_node(self):

        self.stop()

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = AutonomousMotionNode()

    try:

        node.main_loop()

    except KeyboardInterrupt:

        pass

    finally:

        node.stop()
        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
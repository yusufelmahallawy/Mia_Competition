#!/usr/bin/env python3
"""
Autonomous Motion Control - State Machine (ROS2 / rclpy)
Piston Cup Finals - Electrical Training 26/27
Member 2 - Autonomous Motion Control

CONFIRMED: ROS2 Jazzy, with a micro-ROS agent bridging the microcontroller
(publishing /ultrasonic_distance, subscribing /cmd_vel). The camera
(/mono/image) is assumed to come from a separate ROS2 node on a companion
computer, NOT from the same microcontroller - confirm this with the team.

REMAINING ASSUMPTIONS TO CONFIRM BEFORE TESTING:
  1. /ultrasonic_distance is assumed to publish sensor_msgs/Range.
     Check with: `ros2 topic type /ultrasonic_distance`
     If it's something else (e.g. std_msgs/Float32), adjust
     `ultrasonic_callback` and the import accordingly.
  2. /mono/image is assumed to be sensor_msgs/Image, converted with cv_bridge.
     Check with: `ros2 topic type /mono/image`
  3. The real scroll-detection model is NOT included here. `detect_scroll()`
     is a stub - Member 1 (Vision Lead) plugs the trained model in there.
  4. ALL numeric constants in the CONFIG section are PLACEHOLDERS.
     They MUST be replaced with values measured during physical calibration
     (see the "Calibration Checklist" in the algorithm document). Do not
     run this on the real robot with placeholder values.

Testing order:
  1. `ros2 topic list` and `ros2 topic echo /ultrasonic_distance` first,
     to confirm the agent bridge is actually working end to end.
  2. Test each motion primitive alone (move_timed forward, rotate_90, strafe)
     on open ground, and fill in the real numbers in Config.
  3. Test move_forward_until_distance alone (no camera yet).
  4. Once Member 1's model is wired into detect_scroll(), test
     visual_servo_approach() alone.
  5. Only then run the full state machine on the actual field.
"""

import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Range

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None  # allows the file to be imported/reviewed without ROS deps


# ============================================================
# CONFIG - CALIBRATE ALL OF THESE ON THE REAL ROBOT (placeholders!)
# ============================================================
class Config:
    # --- Speeds (m/s and rad/s) ---
    FORWARD_SPEED = 0.15          # TODO calibrate: m/s during row scan
    APPROACH_SPEED = 0.08         # TODO calibrate: slower speed near a scroll
    STRAFE_SPEED = 0.15           # TODO calibrate
    ROTATE_SPEED = 0.6            # TODO calibrate: rad/s during a turn

    # --- Time-based motion (SECONDS) - replace with measured values ---
    TURN_90_DURATION = 1.3            # TODO calibrate with a protractor
    ROW_SCAN_MAX_DURATION = 6.0       # TODO calibrate: time to cross one lane
    RETURN_TIMEOUT_FAILSAFE = 8.0     # TODO calibrate: generous fail-safe only,
                                       # NOT the primary stop condition

    # --- Distance thresholds (METERS) ---
    SAFETY_STOP_DISTANCE = 0.12       # emergency stop, always active
    APPROACH_STOP_DISTANCE_FAR = 0.40 # stop distance for first-scroll approach
    APPROACH_STOP_DISTANCE_NEAR = 0.30# stop distance for second-scroll approach
    CANDIDATE_OBSTACLE_DISTANCE = 2.05  # TODO verify against real field measurement
    KNOWN_WALL_DISTANCE = 1.50        # TODO measure: distance start->far wall,
                                       # used as the PRIMARY stop condition when
                                       # returning toward the start point

    # --- Detection retry policy ---
    MAX_DETECTION_RETRIES = 3
    RETRY_BACKUP_DISTANCE = 0.08      # meters to back up between retries

    # --- Control loop ---
    LOOP_DT = 0.05  # seconds (~20 Hz)


class State(Enum):
    INIT = auto()
    CHECK_INITIAL_PATH = auto()
    # Branch A: path clear
    A_ROTATE_TO_ROW = auto()
    A_SCAN_ROW = auto()
    A_APPROACH_1 = auto()
    A_DETECT_1 = auto()
    A_RESUME_ROW = auto()
    A_APPROACH_2 = auto()
    A_DETECT_2 = auto()
    # Branch B: obstacle directly ahead
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


class AutonomousStateMachine(Node):
    def __init__(self):
        super().__init__("autonomous_state_machine")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_subscription(Range, "/ultrasonic_distance",
                                  self.ultrasonic_callback, 10)
        self.create_subscription(Image, "/mono/image",
                                  self.image_callback, 10)

        self.bridge = CvBridge() if CvBridge else None
        self.latest_range_m = None      # meters
        self.latest_cv_image = None
        self.scrolls_found = 0
        self.state = State.INIT

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------
    def ultrasonic_callback(self, msg: Range):
        self.latest_range_m = msg.range

    def image_callback(self, msg: Image):
        if self.bridge is None:
            return
        self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # ------------------------------------------------------------------
    # ROS2 loop helper: process callbacks, then sleep.
    # (A simple pattern for a blocking state machine under rclpy; fine for
    # bring-up testing. If this grows more complex, consider migrating to
    # a timer-callback-driven design instead.)
    # ------------------------------------------------------------------
    def spin_and_sleep(self, dt=None):
        rclpy.spin_once(self, timeout_sec=0.0)
        time.sleep(dt if dt is not None else Config.LOOP_DT)

    # ------------------------------------------------------------------
    # Low-level motion primitives (all time-based - no encoder feedback)
    # ------------------------------------------------------------------
    def publish_twist(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y  # strafe component (mecanum)
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def stop(self):
        self.publish_twist(0.0, 0.0, 0.0)

    def is_safety_triggered(self):
        return (
            self.latest_range_m is not None
            and self.latest_range_m < Config.SAFETY_STOP_DISTANCE
        )

    def move_timed(self, linear_x=0.0, linear_y=0.0, angular_z=0.0,
                    duration=0.0, stop_on_safety=True):
        """
        Blocking, time-based motion primitive. Safety check runs continuously.
        Returns True if it completed the full duration, False if interrupted.
        """
        start = time.time()
        while time.time() - start < duration and rclpy.ok():
            if stop_on_safety and self.is_safety_triggered():
                self.stop()
                self.get_logger().warn("Safety stop triggered during move_timed()")
                return False
            self.publish_twist(linear_x, linear_y, angular_z)
            self.spin_and_sleep()
        self.stop()
        return True

    def rotate_90(self, direction=1):
        """direction: +1 = left turn, -1 = right turn (adjust sign to match robot)."""
        return self.move_timed(
            angular_z=Config.ROTATE_SPEED * direction,
            duration=Config.TURN_90_DURATION,
        )

    def move_forward_until_distance(self, target_distance_m, max_duration,
                                     speed=None):
        """
        PRIMARY stop condition = ultrasonic reading reaching target_distance_m.
        max_duration is a fail-safe ONLY: if it triggers, treat as a
        stuck/lost condition rather than a successful arrival.
        """
        speed = speed or Config.FORWARD_SPEED
        start = time.time()
        while rclpy.ok():
            if self.is_safety_triggered():
                self.stop()
                return "safety_stop"
            if time.time() - start > max_duration:
                self.stop()
                self.get_logger().warn(
                    "Fail-safe timeout reached in move_forward_until_distance()")
                return "timeout_failsafe"
            if (
                self.latest_range_m is not None
                and self.latest_range_m <= target_distance_m
            ):
                self.stop()
                return "reached_target"
            self.publish_twist(linear_x=speed)
            self.spin_and_sleep()
        return "shutdown"

    # ------------------------------------------------------------------
    # Vision hook - Member 1 plugs the trained model in here
    # ------------------------------------------------------------------
    def detect_scroll(self):
        """
        TODO (Vision Lead): replace with the real model call.
        Must return: (found: bool, bbox: (x, y, w, h) or None, confidence: float)
        bbox is in pixel coordinates of self.latest_cv_image.
        """
        if self.latest_cv_image is None:
            return False, None, 0.0
        # --- placeholder stub ---
        return False, None, 0.0

    def visual_servo_approach(self, stop_distance_m):
        """
        Steers based on where the detected object sits in the frame,
        approaching until either the ultrasonic stop distance is reached
        or the safety layer triggers. Returns True on success.
        """
        while rclpy.ok():
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
            if not found or self.latest_cv_image is None:
                # nothing visible right now - creep forward slowly and keep checking
                self.publish_twist(linear_x=Config.APPROACH_SPEED)
                self.spin_and_sleep()
                continue

            frame_width = self.latest_cv_image.shape[1]
            x, y, w, h = bbox
            box_center_x = x + w / 2.0
            frame_center_x = frame_width / 2.0
            offset = box_center_x - frame_center_x  # >0 => object is right of center

            # TODO calibrate deadband and gain to the actual camera FOV
            deadband_px = frame_width * 0.08
            lateral_gain = 0.15

            if abs(offset) < deadband_px:
                self.publish_twist(linear_x=Config.APPROACH_SPEED)
            else:
                strafe = lateral_gain if offset > 0 else -lateral_gain
                self.publish_twist(linear_x=Config.APPROACH_SPEED * 0.5,
                                    linear_y=strafe)
            self.spin_and_sleep()
        return False

    def attempt_detection_with_retry(self):
        for attempt in range(Config.MAX_DETECTION_RETRIES):
            found, bbox, confidence = self.detect_scroll()
            if found:
                return True
            self.get_logger().info(f"Detection attempt {attempt + 1} failed, retrying...")
            self.move_timed(linear_x=-Config.APPROACH_SPEED,
                             duration=1.0, stop_on_safety=True)
        self.get_logger().warn("All detection retries exhausted - treating as false positive")
        return False

    # ------------------------------------------------------------------
    # State machine driver
    # ------------------------------------------------------------------
    def signal_done(self):
        # TODO (Integration/Member 5): replace with the actual signal
        # mechanism agreed with the team (LED / buzzer / on-screen message).
        self.get_logger().info("=== AUTONOMOUS PHASE COMPLETE - signaling Runner ===")

    def run(self):
        self.get_logger().info("Waiting for first sensor data...")
        while self.latest_range_m is None and rclpy.ok():
            self.spin_and_sleep()

        self.state = State.CHECK_INITIAL_PATH

        while rclpy.ok() and self.state != State.DONE:

            if self.state == State.CHECK_INITIAL_PATH:
                path_clear = self.latest_range_m > Config.CANDIDATE_OBSTACLE_DISTANCE
                self.state = State.A_ROTATE_TO_ROW if path_clear else State.B_MOVE_TO_OBSTACLE

            # ---------------- Branch A: path clear ----------------
            elif self.state == State.A_ROTATE_TO_ROW:
                self.rotate_90(direction=1)
                self.state = State.A_SCAN_ROW

            elif self.state == State.A_SCAN_ROW:
                found, _, _ = self.detect_scroll()
                if found:
                    self.state = State.A_APPROACH_1
                else:
                    ok = self.move_timed(linear_x=Config.FORWARD_SPEED, duration=0.3)
                    if not ok:
                        self.state = State.STUCK

            elif self.state == State.A_APPROACH_1:
                self.visual_servo_approach(Config.APPROACH_STOP_DISTANCE_FAR)
                self.state = State.A_DETECT_1

            elif self.state == State.A_DETECT_1:
                if self.attempt_detection_with_retry():
                    self.scrolls_found += 1
                    self.state = State.A_RESUME_ROW
                else:
                    self.state = State.A_SCAN_ROW  # ignore false positive, keep scanning

            elif self.state == State.A_RESUME_ROW:
                found, _, _ = self.detect_scroll()
                if found:
                    self.state = State.A_APPROACH_2
                else:
                    self.move_timed(linear_x=Config.FORWARD_SPEED, duration=0.3)

            elif self.state == State.A_APPROACH_2:
                self.visual_servo_approach(Config.APPROACH_STOP_DISTANCE_FAR)
                self.state = State.A_DETECT_2

            elif self.state == State.A_DETECT_2:
                if self.attempt_detection_with_retry():
                    self.scrolls_found += 1
                    self.state = State.DONE
                else:
                    self.state = State.A_RESUME_ROW

            # ---------------- Branch B: obstacle directly ahead ----------------
            elif self.state == State.B_MOVE_TO_OBSTACLE:
                self.move_forward_until_distance(
                    Config.APPROACH_STOP_DISTANCE_FAR,
                    max_duration=Config.ROW_SCAN_MAX_DURATION,
                )
                self.state = State.B_DETECT_1

            elif self.state == State.B_DETECT_1:
                if self.attempt_detection_with_retry():
                    self.scrolls_found += 1
                    self.state = State.B_ROTATE_TO_RETURN
                else:
                    self.state = State.STUCK

            elif self.state == State.B_ROTATE_TO_RETURN:
                # Golden rule fix: rotate so the sensor actually faces the
                # direction of travel back toward the start point.
                self.rotate_90(direction=-1)
                self.state = State.B_RETURN_TO_START

            elif self.state == State.B_RETURN_TO_START:
                result = self.move_forward_until_distance(
                    Config.KNOWN_WALL_DISTANCE,
                    max_duration=Config.RETURN_TIMEOUT_FAILSAFE,
                )
                if result == "timeout_failsafe":
                    self.get_logger().warn(
                        "Return-to-start hit fail-safe timeout - flag for Runner")
                    self.state = State.STUCK
                else:
                    self.state = State.B_ROTATE_TO_SEARCH2

            elif self.state == State.B_ROTATE_TO_SEARCH2:
                self.rotate_90(direction=1)
                self.state = State.B_SCAN_FOR_CANDIDATE

            elif self.state == State.B_SCAN_FOR_CANDIDATE:
                if (
                    self.latest_range_m is not None
                    and self.latest_range_m < Config.CANDIDATE_OBSTACLE_DISTANCE
                ):
                    self.state = State.B_APPROACH_2
                else:
                    ok = self.move_timed(linear_x=Config.FORWARD_SPEED, duration=0.3)
                    if not ok:
                        self.state = State.STUCK

            elif self.state == State.B_APPROACH_2:
                self.visual_servo_approach(Config.APPROACH_STOP_DISTANCE_NEAR)
                self.state = State.B_DETECT_2

            elif self.state == State.B_DETECT_2:
                if self.attempt_detection_with_retry():
                    self.scrolls_found += 1
                    self.state = State.DONE
                else:
                    self.state = State.B_SCAN_FOR_CANDIDATE

            elif self.state == State.STUCK:
                self.get_logger().error(
                    "Robot flagged STUCK - waiting for Runner to request a reset")
                self.stop()
                time.sleep(1.0)
                # Stays here until externally reset (Runner/judge action).

            self.spin_and_sleep()

        if self.state == State.DONE:
            self.signal_done()


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousStateMachine()
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

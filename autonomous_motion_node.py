#!/usr/bin/env python3
"""
autonomous_motion_node.py

Piston Cup Finals - Electrical Training 26/27
Member 2 - Autonomous Motion Control
Algorithm version: v5 (search-and-detect with front ultrasonic only,
no encoder, no IMU - all confirmed with the team)

Publishes:
    /auto_cmd_vel        (geometry_msgs/Twist)   - NOT /cmd_vel directly.
                                                    Integration Node forwards
                                                    this to /cmd_vel.
    /scroll_count         (std_msgs/Int32)
    /autonomous_debug_state (std_msgs/String)     - current state name +
                                                     latest ultrasonic reading,
                                                     for live debugging during
                                                     test runs.

Subscribes:
    /ultrasonic_distance  (sensor_msgs/Range)
    /mono/image           (sensor_msgs/Image)

Services:
    /start_autonomous  (std_srvs/Trigger)
    /reset_autonomous  (std_srvs/Trigger)

IMPORTANT - confirmed assumptions (do not change without re-checking):
    - ROS2 Jazzy, micro-ROS agent bridges the ultrasonic + cmd_vel side.
    - This node does NOT publish to /cmd_vel directly - Integration Node
      is the only thing allowed to touch /cmd_vel.
    - No encoder, no IMU. Every rotation and every strafe is TIME-BASED
      and must be calibrated by hand (see Config below).
    - Forward/backward motion along the sensor's own facing direction IS
      controlled by real ultrasonic feedback (that direction is trustworthy).
      Strafing (sideways) and rotating are blind to the sensor - this is a
      known, accepted risk given the hardware, not an oversight. Keep
      strafe/rotate durations conservative.

KNOWN OPEN RISK (flagged, not solved): the emergency-stop recovery
("return to start and restart the whole algorithm") is itself a blind,
time-based maneuver, executed at the worst possible moment (right after
something unexpected triggered the stop). This is the team's explicit
decision given the hardware constraints - documented here so it isn't
forgotten, not silently accepted.
"""

import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None


# ============================================================
# CONFIG - ALL PLACEHOLDERS. CALIBRATE ON THE REAL ROBOT.
# ============================================================
class Config:

    # --- Speeds ---
    FORWARD_SPEED = 0.15
    APPROACH_SPEED = 0.08
    STRAFE_SPEED = 0.15
    ROTATE_SPEED = 0.6

    # --- Distance thresholds (METERS, measured by ultrasonic) ---
    # Nothing close ahead at all -> need to search sideways.
    FRONT_FAR_THRESHOLD = 2.00

    # After the CCW 90 deg rotate (facing the centre divider wall):
    # measured baseline ~1.00 m clear, ~0.65 m with a scroll against the
    # wall, 0.60 m used as the "definitely something there" cutoff
    # (see the team's own hand calculation - re-verify against the real
    # robot's front offset before trusting this number).
    DIVIDER_OBSTACLE_THRESHOLD = 0.60

    # Stop distance to attempt a detection.
    DETECT_APPROACH_DISTANCE = 0.20

    # Emergency stop - MUST stay clearly smaller than DETECT_APPROACH_DISTANCE,
    # otherwise every intentional approach to 20 cm would also trigger the
    # emergency layer. Keep at least 10 cm of margin between the two.
    EMERGENCY_STOP_DISTANCE = 0.08

    # --- Time-based motion (SECONDS) - replace with measured values ---
    STRAFE_SEARCH_STEP_DURATION = 1.0   # TODO calibrate: ~20 cm strafe left
    ROW_SWEEP_MAX_DURATION = 6.0        # TODO calibrate: ~280 cm strafe right
    TURN_90_DURATION = 1.3              # TODO calibrate with a protractor
    POST_STRAFE_WAIT = 3.0              # wait before re-scanning after a strafe step
    POST_SUCCESS_WAIT = 5.0             # cool-down after a successful detection
    EMERGENCY_RECOVERY_DURATION = 2.0   # TODO calibrate: blind reverse-away time

    # --- Detection ---
    DETECT_FEEDBACK_TIMEOUT = 5.0       # seconds to wait for model feedback
    MAX_DETECTION_RETRIES = 3           # in-place retries before giving up
    RETRY_WAIT = 1.0

    LOOP_DT = 0.05


class State(Enum):
    WAITING = auto()
    SCAN_FRONT = auto()
    STRAFE_SEARCH = auto()
    SCAN_AFTER_STRAFE = auto()
    ROTATE_CCW = auto()
    SCAN_AFTER_ROTATE = auto()
    SWEEP_ROW = auto()
    APPROACH_DIRECT = auto()
    ATTEMPT_DETECT = auto()
    RETREAT_AFTER_SUCCESS = auto()
    RETREAT_AFTER_FAIL = auto()
    EMERGENCY = auto()
    DONE = auto()
    STUCK = auto()


class AutonomousMotionNode(Node):

    def __init__(self):
        super().__init__("autonomous_motion_node")

        # ---------------- Publishers ----------------
        self.cmd_pub = self.create_publisher(Twist, "/auto_cmd_vel", 10)
        self.scroll_count_pub = self.create_publisher(Int32, "/scroll_count", 10)
        self.debug_pub = self.create_publisher(String, "/autonomous_debug_state", 10)

        # ---------------- Subscribers ----------------
        self.create_subscription(Range, "/ultrasonic_distance",
                                  self.ultrasonic_callback, 10)
        self.create_subscription(Image, "/mono/image",
                                  self.image_callback, 10)

        # ---------------- Services ----------------
        self.create_service(Trigger, "/start_autonomous",
                             self.start_autonomous_callback)
        self.create_service(Trigger, "/reset_autonomous",
                             self.reset_autonomous_callback)

        # ---------------- State ----------------
        self.bridge = CvBridge() if CvBridge else None
        self.latest_range_m = None
        self.latest_cv_image = None
        self.scrolls_found = 0
        self.state = State.WAITING
        self.autonomous_enabled = False
        self.saved_dist = None

        self.get_logger().info("Autonomous Motion Node ready")

    # ========================================================
    # SENSOR CALLBACKS
    # ========================================================
    def ultrasonic_callback(self, msg):
        self.latest_range_m = msg.range

    def image_callback(self, msg):
        if self.bridge is None:
            return
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Camera conversion error: {e}")

    # ========================================================
    # START / RESET SERVICES
    # ========================================================
    def start_autonomous_callback(self, request, response):
        if self.autonomous_enabled:
            response.success = False
            response.message = "Autonomous already running"
            return response

        self.scrolls_found = 0
        self.publish_scroll_count()
        self.saved_dist = None
        self.set_state(State.SCAN_FRONT)
        self.autonomous_enabled = True

        self.get_logger().info("AUTONOMOUS STARTED")
        response.success = True
        response.message = "Autonomous started"
        return response

    def reset_autonomous_callback(self, request, response):
        self.stop()
        self.autonomous_enabled = False
        self.scrolls_found = 0
        self.saved_dist = None
        self.set_state(State.WAITING)
        self.publish_scroll_count()

        self.get_logger().warn("AUTONOMOUS RESET (manual)")
        response.success = True
        response.message = "Autonomous reset"
        return response

    # ========================================================
    # DEBUG / LOGGING - required by the team for test-run debugging
    # ========================================================
    def set_state(self, new_state):
        self.state = new_state
        self.log_state()

    def log_state(self):
        reading = "n/a" if self.latest_range_m is None else f"{self.latest_range_m:.3f} m"
        msg = f"[STATE] {self.state.name} | ultrasonic = {reading}"
        self.get_logger().info(msg)
        out = String()
        out.data = msg
        self.debug_pub.publish(out)

    # ========================================================
    # ROS2 LOOP HELPER
    # ========================================================
    def spin_and_sleep(self, dt=None):
        rclpy.spin_once(self, timeout_sec=0.0)
        time.sleep(dt if dt is not None else Config.LOOP_DT)

    # ========================================================
    # LOW-LEVEL MOTION
    # ========================================================
    def publish_twist(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = linear_y
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish_twist(0.0, 0.0, 0.0)

    def is_emergency_triggered(self):
        return (
            self.latest_range_m is not None
            and self.latest_range_m < Config.EMERGENCY_STOP_DISTANCE
        )

    def move_timed(self, linear_x=0.0, linear_y=0.0, angular_z=0.0,
                    duration=0.0, check_emergency=True):
        """
        Blind, time-based motion. Returns "completed", "emergency", or
        "disabled" (autonomous turned off / node shutting down mid-move).
        """
        start = time.time()
        while time.time() - start < duration and rclpy.ok() and self.autonomous_enabled:
            if check_emergency and self.is_emergency_triggered():
                self.stop()
                return "emergency"
            self.publish_twist(linear_x, linear_y, angular_z)
            self.spin_and_sleep()
        self.stop()
        return "completed" if self.autonomous_enabled else "disabled"

    def move_forward_until(self, target_distance_m, max_duration, speed=None,
                            direction=1):
        """
        Sensor-guided motion along the direction the ultrasonic is actually
        facing (forward = direction=1, backward = direction=-1). Since the
        sensor faces forward, "backward" here relies on the assumption that
        moving away from the target only ever increases the reading - true
        for a straight-line retreat along the same axis just approached.
        Returns "reached_target", "emergency", "timeout_failsafe", or "disabled".
        """
        speed = speed or Config.FORWARD_SPEED
        start = time.time()
        while rclpy.ok() and self.autonomous_enabled:
            if self.is_emergency_triggered():
                self.stop()
                return "emergency"
            if time.time() - start > max_duration:
                self.stop()
                self.get_logger().warn("move_forward_until: fail-safe timeout")
                return "timeout_failsafe"
            if self.latest_range_m is not None:
                if direction > 0 and self.latest_range_m <= target_distance_m:
                    self.stop()
                    return "reached_target"
                if direction < 0 and self.latest_range_m >= target_distance_m:
                    self.stop()
                    return "reached_target"
            self.publish_twist(linear_x=speed * direction)
            self.spin_and_sleep()
        self.stop()
        return "disabled"

    # ========================================================
    # VISION HOOK - Member 1 plugs the trained model in here
    # ========================================================
    def detect_scroll(self):
        """
        TODO (Vision Lead): replace with the real model call.
        Returns: found: bool, bbox: (x, y, w, h) or None, confidence: float
        """
        if self.latest_cv_image is None:
            return False, None, 0.0
        return False, None, 0.0

    def wait_for_detection(self, timeout_s):
        """Poll detect_scroll() until it reports found=True or timeout."""
        start = time.time()
        while time.time() - start < timeout_s and rclpy.ok() and self.autonomous_enabled:
            if self.is_emergency_triggered():
                self.stop()
                return "emergency"
            found, bbox, conf = self.detect_scroll()
            if found:
                return "success"
            self.spin_and_sleep()
        return "timeout" if self.autonomous_enabled else "disabled"

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
        self.get_logger().info(f"Scroll detected: {self.scrolls_found}")

    # ========================================================
    # STATE MACHINE
    # ========================================================
    def run(self):
        while rclpy.ok() and self.autonomous_enabled:

            if self.is_emergency_triggered() and self.state not in (
                State.EMERGENCY, State.WAITING, State.DONE
            ):
                self.set_state(State.EMERGENCY)

            # ---------------- INITIAL SCAN ----------------
            if self.state == State.SCAN_FRONT:
                if self.latest_range_m is None:
                    self.spin_and_sleep()
                    continue
                if self.latest_range_m >= Config.FRONT_FAR_THRESHOLD:
                    self.set_state(State.STRAFE_SEARCH)
                else:
                    self.set_state(State.APPROACH_DIRECT)

            elif self.state == State.STRAFE_SEARCH:
                result = self.move_timed(linear_y=Config.STRAFE_SPEED,
                                          duration=Config.STRAFE_SEARCH_STEP_DURATION)
                if result != "completed":
                    continue  # emergency/disabled handled at top of loop
                self.move_timed(duration=Config.POST_STRAFE_WAIT, check_emergency=False)
                self.set_state(State.SCAN_AFTER_STRAFE)

            elif self.state == State.SCAN_AFTER_STRAFE:
                if self.latest_range_m is None:
                    self.spin_and_sleep()
                    continue
                if self.latest_range_m >= Config.FRONT_FAR_THRESHOLD:
                    self.set_state(State.ROTATE_CCW)
                else:
                    self.set_state(State.APPROACH_DIRECT)

            elif self.state == State.ROTATE_CCW:
                result = self.move_timed(angular_z=Config.ROTATE_SPEED,
                                          duration=Config.TURN_90_DURATION)
                if result != "completed":
                    continue
                self.set_state(State.SCAN_AFTER_ROTATE)

            elif self.state == State.SCAN_AFTER_ROTATE:
                if self.latest_range_m is None:
                    self.spin_and_sleep()
                    continue
                if self.latest_range_m >= Config.DIVIDER_OBSTACLE_THRESHOLD:
                    self.set_state(State.SWEEP_ROW)
                else:
                    self.set_state(State.APPROACH_DIRECT)

            elif self.state == State.SWEEP_ROW:
                # Strafe right, watching continuously for a candidate obstacle.
                start = time.time()
                found_candidate = False
                while (time.time() - start < Config.ROW_SWEEP_MAX_DURATION
                       and rclpy.ok() and self.autonomous_enabled):
                    if self.is_emergency_triggered():
                        self.stop()
                        break
                    if (self.latest_range_m is not None
                            and self.latest_range_m < Config.DIVIDER_OBSTACLE_THRESHOLD):
                        self.stop()
                        found_candidate = True
                        break
                    self.publish_twist(linear_y=-Config.STRAFE_SPEED)
                    self.spin_and_sleep()
                self.stop()
                if not self.autonomous_enabled:
                    continue
                self.set_state(State.APPROACH_DIRECT if found_candidate else State.SCAN_FRONT)

            # ---------------- APPROACH + DETECT ----------------
            elif self.state == State.APPROACH_DIRECT:
                self.saved_dist = self.latest_range_m
                result = self.move_forward_until(
                    Config.DETECT_APPROACH_DISTANCE,
                    max_duration=Config.ROW_SWEEP_MAX_DURATION,
                )
                if result == "reached_target":
                    self.set_state(State.ATTEMPT_DETECT)
                elif result == "emergency":
                    continue  # handled at top of loop
                elif result == "timeout_failsafe":
                    self.get_logger().warn("Approach timed out before reaching detect distance")
                    self.set_state(State.STUCK)
                # "disabled" falls through and the outer while exits naturally

            elif self.state == State.ATTEMPT_DETECT:
                outcome = self.wait_for_detection(Config.DETECT_FEEDBACK_TIMEOUT)
                if outcome == "success":
                    self.increment_scroll_count()
                    self.set_state(State.RETREAT_AFTER_SUCCESS)
                elif outcome == "timeout":
                    self.retry_or_give_up()
                elif outcome == "emergency":
                    continue

            elif self.state == State.RETREAT_AFTER_SUCCESS:
                target = self.saved_dist if self.saved_dist is not None else 0.5
                result = self.move_forward_until(
                    target, max_duration=Config.ROW_SWEEP_MAX_DURATION, direction=-1
                )
                if result == "emergency":
                    continue
                self.move_timed(duration=Config.POST_SUCCESS_WAIT, check_emergency=False)
                if self.scrolls_found >= 2:
                    self.set_state(State.DONE)
                else:
                    self.set_state(State.SCAN_FRONT)

            elif self.state == State.RETREAT_AFTER_FAIL:
                target = self.saved_dist if self.saved_dist is not None else 0.5
                result = self.move_forward_until(
                    target, max_duration=Config.ROW_SWEEP_MAX_DURATION, direction=-1
                )
                if result == "emergency":
                    continue
                self.set_state(State.SCAN_FRONT)

            # ---------------- EMERGENCY ----------------
            elif self.state == State.EMERGENCY:
                self.get_logger().error(
                    "EMERGENCY STOP triggered - attempting blind recovery to start "
                    "and restarting the whole algorithm (known risk: this recovery "
                    "move is itself blind/time-based)"
                )
                self.stop()
                # Best-effort: back away from whatever tripped the sensor.
                self.move_timed(linear_x=-Config.FORWARD_SPEED,
                                 duration=Config.EMERGENCY_RECOVERY_DURATION,
                                 check_emergency=False)
                self.scrolls_found = 0
                self.saved_dist = None
                self.publish_scroll_count()
                self.set_state(State.SCAN_FRONT)

            # ---------------- TERMINAL STATES ----------------
            elif self.state == State.DONE:
                self.stop()
                self.get_logger().info("AUTONOMOUS PHASE COMPLETE")
                self.get_logger().info(f"Scrolls detected: {self.scrolls_found}")
                self.autonomous_enabled = False
                break

            elif self.state == State.STUCK:
                self.stop()
                self.get_logger().error("ROBOT STUCK - waiting for /reset_autonomous")
                self.autonomous_enabled = False
                break

            self.spin_and_sleep()

    def retry_or_give_up(self):
        """Bounded in-place retry after a detection timeout."""
        if not hasattr(self, "_retry_count"):
            self._retry_count = 0
        self._retry_count += 1
        if self._retry_count < Config.MAX_DETECTION_RETRIES:
            self.get_logger().info(
                f"Detection timeout - retry {self._retry_count}/{Config.MAX_DETECTION_RETRIES}"
            )
            self.move_timed(duration=Config.RETRY_WAIT, check_emergency=False)
            self.set_state(State.ATTEMPT_DETECT)
        else:
            self._retry_count = 0
            self.get_logger().warn("Detection retries exhausted - treating as missed candidate")
            self.set_state(State.RETREAT_AFTER_FAIL)

    # ========================================================
    # MAIN LOOP
    # ========================================================
    def main_loop(self):
        while rclpy.ok():
            if not self.autonomous_enabled:
                self.stop()
                rclpy.spin_once(self, timeout_sec=0.1)
                continue
            self.run()


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

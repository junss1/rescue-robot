#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# explore_detect_supervisor
# - /robot5/start True 전까지 대기
# - frontier 탐색 중 일정 거리마다 explore_lite pause -> step scan
# - exploration_complete / returning_to_origin 이 오면 supervisor가 직접 initial pose로 복귀
# - HOME_DONE 되면 /robot5/mission_completed True 1회 발행
# - HOME_DONE 후 /robot5/dock action 수행
# - 어떤 동작 중이든 /robot/stop True 가 오면 즉시 initial pose 복귀 후 dock 수행
# -----------------------------------------------------------------------------

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from explore_lite_msgs.msg import ExploreStatus
from irobot_create_msgs.action import Dock


def norm_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )


class ExploreDetectSupervisor(Node):
    def __init__(self):
        super().__init__('explore_detect_supervisor')

        # =========================
        # start / stop / mission
        # =========================
        self.start_requested = False
        self.stop_requested = False
        self.start_wait_log_t = 0.0
        self.mission_completed_sent = False

        # =========================
        # scan config
        # =========================
        self.scan_every_odom_m = 1.0
        self.scan_turn_wz = 0.40
        self.scan_step_rad = math.radians(30.0)
        self.scan_total_rad = 2.0 * math.pi
        self.scan_finish_margin = math.radians(2.0)
        self.scan_settle_sec = 0.20
        self.scan_dwell_sec = 2.00
        self.scan_detect_delay_sec = 0.50
        self.scan_stop_wait_timeout = 1.5

        # =========================
        # stop thresholds
        # =========================
        self.stop_lin_thresh = 0.06
        self.stop_ang_thresh = 0.15

        # =========================
        # return-home config
        # =========================
        self.base_frames = ['base_link', 'base_footprint']
        self.initial_pose_map = None
        self.initial_pose_captured = False
        self.return_retry_count = 0
        self.max_return_retries = 3
        self.return_retry_wait_sec = 1.0
        self.last_return_attempt_t = 0.0

        # =========================
        # dock config
        # =========================
        self.dock_goal_future = None
        self.dock_goal_handle = None
        self.dock_result_future = None
        self.dock_action_sent = False
        self.dock_result_logged = False

        # =========================
        # state
        # =========================
        self.state = 'WAIT_EXPLORE'
        self.state_started_t = time.time()
        self.last_wait_log_t = 0.0

        self.explore_ready = False
        self.explore_status = ''

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None
        self.lin_speed = 0.0
        self.ang_speed = 0.0
        self.last_odom_xy = None
        self.explore_travel_dist = 0.0

        self.scan_remaining = 0.0
        self.scan_step_start_yaw = None
        self.scan_step_target = 0.0

        ns = self.get_namespace().rstrip('/')
        self.odom_topic = f'{ns}/odom' if ns else '/odom'
        self.cmd_vel_topic = 'cmd_vel'
        self.explore_resume_topic = 'explore/resume'
        self.explore_status_topic = 'explore/status'
        self.scan_active_topic = 'scan_active'
        self.start_topic = 'start'
        self.stop_topic = '/robot/stop'          # 절대 토픽
        self.mission_completed_topic = 'mission_completed'

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Nav2
        nav_ns = self.get_namespace().strip('/')
        self.nav_navigator = BasicNavigator(
            namespace=nav_ns,
            node_name='robot5_nav_navigator'
        )
        self.get_logger().info(f'Waiting for bt_navigator active in namespace: {nav_ns}')
        self.nav_navigator._waitForNodeToActivate('bt_navigator')
        self.get_logger().info('bt_navigator active.')

        # Dock action client (relative name -> /robot5/dock)
        self.dock_client = ActionClient(self, Dock, 'dock')

        # subs / pubs
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(ExploreStatus, self.explore_status_topic, self.explore_status_cb, 10)
        self.create_subscription(Bool, self.start_topic, self.start_cb, 10)
        self.create_subscription(Bool, self.stop_topic, self.stop_cb, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.explore_resume_pub = self.create_publisher(Bool, self.explore_resume_topic, 10)
        self.scan_active_pub = self.create_publisher(Bool, self.scan_active_topic, 10)
        self.mission_completed_pub = self.create_publisher(Bool, self.mission_completed_topic, 10)

        self.publish_scan_active(False)
        self.timer = self.create_timer(0.05, self.step)

        self.get_logger().info('explore_detect_supervisor started')

    # ------------------------------------------------------------------
    # status helpers
    # ------------------------------------------------------------------
    def is_scan_allowed_status(self) -> bool:
        return self.explore_status in {
            ExploreStatus.EXPLORATION_STARTED,
            ExploreStatus.EXPLORATION_IN_PROGRESS,
        }

    def is_return_trigger_status(self) -> bool:
        return self.explore_status in {
            ExploreStatus.EXPLORATION_COMPLETE,
            ExploreStatus.RETURNING_TO_ORIGIN,
        }

    def is_return_done_status(self) -> bool:
        return self.explore_status == ExploreStatus.RETURNED_TO_ORIGIN

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.robot_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        wz = msg.twist.twist.angular.z
        self.lin_speed = math.hypot(vx, vy)
        self.ang_speed = abs(wz)

        if self.last_odom_xy is None:
            self.last_odom_xy = (self.robot_x, self.robot_y)
            return

        dx = self.robot_x - self.last_odom_xy[0]
        dy = self.robot_y - self.last_odom_xy[1]
        step_dist = math.hypot(dx, dy)

        if self.state == 'EXPLORE' and self.is_scan_allowed_status():
            self.explore_travel_dist += step_dist

        self.last_odom_xy = (self.robot_x, self.robot_y)

    def explore_status_cb(self, msg: ExploreStatus):
        prev = self.explore_status
        self.explore_status = msg.status

        if not self.explore_ready:
            self.explore_ready = True
            self.get_logger().info(f'explore_lite detected: {msg.status}')

            if self.start_requested:
                self.publish_explore_resume(True)
                self.set_state('EXPLORE')
            else:
                self.publish_explore_resume(False)
                self.set_state('WAIT_START')

        elif prev != msg.status:
            self.get_logger().info(f'explore/status: {msg.status}')

    def start_cb(self, msg: Bool):
        if not msg.data or self.start_requested:
            return

        self.start_requested = True
        self.stop_requested = False
        self.mission_completed_sent = False
        self.get_logger().info('start=True received')

        if self.explore_ready and self.state in ['WAIT_EXPLORE', 'WAIT_START']:
            self.publish_explore_resume(True)
            self.set_state('EXPLORE')

    def stop_cb(self, msg: Bool):
        if not msg.data:
            return
        self.get_logger().warn('/robot/stop=True received -> return home and dock')
        self.stop_requested = True

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def set_state(self, new_state: str):
        if self.state != new_state:
            self.get_logger().info(f'state: {self.state} -> {new_state}')
        self.state = new_state
        self.state_started_t = time.time()

    def check_explore_ready(self):
        if self.explore_resume_pub.get_subscription_count() > 0:
            if not self.explore_ready:
                self.explore_ready = True
                self.get_logger().info('explore_lite detected via /explore/resume subscriber')

                if self.start_requested:
                    self.publish_explore_resume(True)
                    self.set_state('EXPLORE')
                else:
                    self.publish_explore_resume(False)
                    self.set_state('WAIT_START')
        return self.explore_ready

    def publish_explore_resume(self, resume: bool):
        if not self.check_explore_ready():
            return False
        msg = Bool()
        msg.data = bool(resume)
        self.explore_resume_pub.publish(msg)
        return True

    def publish_scan_active(self, active: bool):
        msg = Bool()
        msg.data = bool(active)
        self.scan_active_pub.publish(msg)

    def publish_cmd_vel(self, wz: float):
        msg = Twist()
        msg.angular.z = float(wz)
        self.cmd_vel_pub.publish(msg)

    def publish_mission_completed(self):
        if self.mission_completed_sent:
            return
        msg = Bool()
        msg.data = True
        self.mission_completed_pub.publish(msg)
        self.mission_completed_sent = True
        self.get_logger().info('mission_completed=True published')

    def stop_robot(self):
        self.publish_cmd_vel(0.0)

    def robot_pose_ready(self):
        return self.robot_x is not None and self.robot_y is not None and self.robot_yaw is not None

    def robot_stopped(self):
        return self.lin_speed <= self.stop_lin_thresh and self.ang_speed <= self.stop_ang_thresh

    def reset_scan_context(self):
        self.publish_scan_active(False)
        self.scan_remaining = 0.0
        self.scan_step_start_yaw = None
        self.scan_step_target = 0.0
        self.explore_travel_dist = 0.0

    def reset_dock_context(self):
        self.dock_goal_future = None
        self.dock_goal_handle = None
        self.dock_result_future = None
        self.dock_action_sent = False
        self.dock_result_logged = False

    def cancel_dock_if_any(self):
        try:
            if self.dock_goal_handle is not None:
                self.dock_goal_handle.cancel_goal_async()
        except Exception:
            pass
        self.reset_dock_context()

    def begin_scan(self):
        self.scan_remaining = self.scan_total_rad
        self.scan_step_start_yaw = None
        self.scan_step_target = 0.0
        self.publish_scan_active(False)
        self.set_state('SCAN_PREPARE_STOP')

    def start_next_scan_step(self):
        if not self.robot_pose_ready():
            return

        if self.scan_remaining <= self.scan_finish_margin:
            self.finish_scan_and_resume()
            return

        self.scan_step_start_yaw = self.robot_yaw
        self.scan_step_target = min(self.scan_step_rad, self.scan_remaining)
        self.publish_scan_active(False)
        self.set_state('SCAN_TURN')

    def finish_scan_and_resume(self):
        self.stop_robot()
        self.publish_scan_active(False)
        self.explore_travel_dist = 0.0
        self.scan_remaining = 0.0
        self.scan_step_start_yaw = None
        self.scan_step_target = 0.0
        self.publish_explore_resume(True)
        self.set_state('EXPLORE')

    def maybe_start_progress_scan(self):
        if not self.explore_ready:
            return
        if not self.start_requested:
            return
        if not self.is_scan_allowed_status():
            return
        if not self.robot_pose_ready():
            return
        if self.explore_travel_dist < self.scan_every_odom_m:
            return

        self.get_logger().info(f'start scan by odom dist: {self.explore_travel_dist:.2f} m')
        ok = self.publish_explore_resume(False)
        if not ok:
            self.set_state('WAIT_EXPLORE')
            return
        self.begin_scan()

    # ------------------------------------------------------------------
    # initial pose / return-home
    # ------------------------------------------------------------------
    def get_robot_pose_in_map(self):
        for base_frame in self.base_frames:
            try:
                tf_robot = self.tf_buffer.lookup_transform(
                    'map',
                    base_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )

                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = 'map'
                pose.pose.position.x = tf_robot.transform.translation.x
                pose.pose.position.y = tf_robot.transform.translation.y
                pose.pose.position.z = tf_robot.transform.translation.z
                pose.pose.orientation = tf_robot.transform.rotation
                return pose
            except Exception:
                continue
        return None

    def maybe_capture_initial_pose(self):
        if self.initial_pose_captured:
            return True

        pose = self.get_robot_pose_in_map()
        if pose is None:
            return False

        self.initial_pose_map = pose
        self.initial_pose_captured = True
        yaw = quaternion_to_yaw(
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )
        self.get_logger().info(
            f'initial pose captured in map: '
            f'({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f}), yaw={yaw:.2f}'
        )
        return True

    def start_return_home(self, reason: str):
        if self.state == 'RETURN_HOME':
            return
        if self.state in ['DOCKING', 'DOCKED']:
            self.cancel_dock_if_any()

        if not self.initial_pose_captured or self.initial_pose_map is None:
            ok = self.maybe_capture_initial_pose()
            if not ok:
                self.get_logger().warn('initial pose not ready yet -> cannot return home')
                return

        self.get_logger().info(f'start return home by supervisor: {reason}')

        self.publish_explore_resume(False)
        self.publish_scan_active(False)
        self.stop_robot()
        self.reset_scan_context()
        self.reset_dock_context()

        try:
            self.nav_navigator.cancelTask()
        except Exception:
            pass

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose = self.initial_pose_map.pose

        self.nav_navigator.goToPose(goal)
        self.last_return_attempt_t = time.time()
        self.return_retry_count = 0
        self.set_state('RETURN_HOME')

    def maybe_trigger_return_home(self):
        if not self.is_return_trigger_status():
            return

        if self.state in ['RETURN_HOME', 'HOME_DONE', 'DOCKING', 'DOCKED']:
            return

        self.start_return_home(reason=self.explore_status)

    def update_return_home(self):
        if self.state != 'RETURN_HOME':
            return

        self.publish_scan_active(False)

        if not self.nav_navigator.isTaskComplete():
            return

        result = self.nav_navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('return home succeeded')
            self.publish_mission_completed()
            self.stop_requested = False
            self.set_state('HOME_DONE')
            return

        self.return_retry_count += 1
        self.get_logger().warn(
            f'return home result={result}, retry={self.return_retry_count}/{self.max_return_retries}'
        )

        if self.return_retry_count > self.max_return_retries:
            self.get_logger().error('return home failed too many times')
            self.set_state('HOME_FAILED')
            return

        if time.time() - self.last_return_attempt_t >= self.return_retry_wait_sec:
            self.start_return_home(reason='retry')

    # ------------------------------------------------------------------
    # docking
    # ------------------------------------------------------------------
    def start_docking(self):
        if self.state == 'DOCKING' or self.state == 'DOCKED':
            return
        if self.dock_action_sent:
            return

        if not self.dock_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('dock action server not available yet')
            return

        self.get_logger().info('sending dock goal')
        goal_msg = Dock.Goal()
        self.dock_goal_future = self.dock_client.send_goal_async(goal_msg)
        self.dock_goal_future.add_done_callback(self.on_dock_goal_response)
        self.dock_action_sent = True
        self.set_state('DOCKING')

    def on_dock_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'dock goal send failed: {e}')
            self.dock_action_sent = False
            self.set_state('DOCK_FAILED')
            return

        if not goal_handle.accepted:
            self.get_logger().error('dock goal rejected')
            self.dock_action_sent = False
            self.set_state('DOCK_FAILED')
            return

        self.get_logger().info('dock goal accepted')
        self.dock_goal_handle = goal_handle
        self.dock_result_future = goal_handle.get_result_async()
        self.dock_result_future.add_done_callback(self.on_dock_result)

    def on_dock_result(self, future):
        try:
            result = future.result()
            status = result.status
            self.get_logger().info(f'dock finished with status={status}')
            self.dock_result_logged = True
            # 4 == STATUS_SUCCEEDED
            if status == 4:
                self.set_state('DOCKED')
            else:
                self.set_state('DOCK_FAILED')
        except Exception as e:
            self.get_logger().error(f'dock result error: {e}')
            self.set_state('DOCK_FAILED')

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def step(self):
        now = time.time()
        self.check_explore_ready()
        self.maybe_capture_initial_pose()

        # 언제든 stop 오면 home -> dock
        if self.stop_requested and self.state not in ['RETURN_HOME', 'HOME_DONE', 'DOCKING', 'DOCKED']:
            self.start_return_home(reason='stop_topic')
            return

        if self.state == 'WAIT_EXPLORE':
            self.publish_scan_active(False)
            if now - self.last_wait_log_t > 5.0:
                self.get_logger().info('waiting for explore_lite...')
                self.last_wait_log_t = now
            return

        if self.state == 'WAIT_START':
            self.publish_scan_active(False)
            self.publish_explore_resume(False)
            if now - self.start_wait_log_t > 5.0:
                self.get_logger().info('waiting for /start True ...')
                self.start_wait_log_t = now
            return

        if self.is_return_trigger_status():
            self.maybe_trigger_return_home()

        if self.is_return_done_status() and self.state not in ['HOME_DONE', 'DOCKING', 'DOCKED']:
            self.publish_scan_active(False)
            self.publish_mission_completed()
            self.set_state('HOME_DONE')
            return

        if self.state == 'HOME_DONE':
            self.publish_scan_active(False)
            self.start_docking()
            return

        if self.state == 'DOCKING':
            self.publish_scan_active(False)
            return

        if self.state == 'DOCKED':
            self.publish_scan_active(False)
            return

        if self.state == 'DOCK_FAILED':
            self.publish_scan_active(False)
            return

        if self.state == 'HOME_FAILED':
            self.publish_scan_active(False)
            return

        if self.state == 'RETURN_HOME':
            self.update_return_home()
            return

        if self.state == 'EXPLORE':
            self.publish_scan_active(False)
            self.maybe_start_progress_scan()
            return

        if self.state == 'SCAN_PREPARE_STOP':
            if self.stop_requested:
                self.start_return_home(reason='stop_during_scan_prepare')
                return
            if self.is_return_trigger_status():
                self.start_return_home(reason='during_scan_prepare')
                return

            self.publish_scan_active(False)
            self.stop_robot()

            waited = time.time() - self.state_started_t
            if self.robot_stopped() or waited > self.scan_stop_wait_timeout:
                self.start_next_scan_step()
            return

        if self.state == 'SCAN_TURN':
            if self.stop_requested:
                self.start_return_home(reason='stop_during_scan_turn')
                return
            if self.is_return_trigger_status():
                self.start_return_home(reason='during_scan_turn')
                return

            if not self.robot_pose_ready() or self.scan_step_start_yaw is None:
                return

            turned = abs(norm_angle(self.robot_yaw - self.scan_step_start_yaw))
            if turned + self.scan_finish_margin >= self.scan_step_target:
                self.stop_robot()
                self.scan_remaining = max(0.0, self.scan_remaining - self.scan_step_target)
                self.set_state('SCAN_SETTLE')
                return

            self.publish_cmd_vel(self.scan_turn_wz)
            return

        if self.state == 'SCAN_SETTLE':
            if self.stop_requested:
                self.start_return_home(reason='stop_during_scan_settle')
                return
            if self.is_return_trigger_status():
                self.start_return_home(reason='during_scan_settle')
                return

            self.stop_robot()
            self.publish_scan_active(False)

            if time.time() - self.state_started_t >= self.scan_settle_sec:
                self.set_state('SCAN_DWELL')
            return

        if self.state == 'SCAN_DWELL':
            if self.stop_requested:
                self.start_return_home(reason='stop_during_scan_dwell')
                return
            if self.is_return_trigger_status():
                self.start_return_home(reason='during_scan_dwell')
                return

            self.stop_robot()

            dwell_elapsed = time.time() - self.state_started_t
            detect_enabled = dwell_elapsed >= self.scan_detect_delay_sec
            self.publish_scan_active(detect_enabled)

            if dwell_elapsed >= self.scan_dwell_sec:
                self.publish_scan_active(False)
                self.start_next_scan_step()
            return

    def destroy_node(self):
        try:
            self.stop_robot()
            self.publish_scan_active(False)
            self.publish_explore_resume(False)
            self.cancel_dock_if_any()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExploreDetectSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
# rescue_nav_node.py v0.401 2026-03-17
# [이번 버전에서 수정된 사항]
# - robot5 연동 입력을 /robot5/robot_pose_at_detection + /robot5/victim_point 기준으로 재구성
# - stop/timeout/undock/predock/dock 흐름을 new nav 기준으로 레거시 실행 경로에 이식
# - victim_point 부재/지연 시 robot5 pose orientation을 fallback yaw로 사용하도록 보강
# - 현재 런타임 계약(robot5 입력 -> arrived -> control 세션 -> tts_done -> 다음 목표)을 주석으로 명확화

"""
rescue_nav_node.py
==================
TurtleBot B - 요구조자 위치 토픽 수신 → 큐 순차 주행 → 미션 완료 대기 → pre-dock → dock

동작 흐름:
  1. /robot5/robot_pose_at_detection (PoseStamped) 수신 ← robot5 검출 당시 위치
  2. /robot5/victim_point (PointStamped) 를 보조 입력으로 캐시
  3. map 프레임으로 변환 후 goal_queue(FIFO)에 (x, y, yaw) 적재
     - 위치: robot5 검출 당시 위치
     - yaw: victim_point를 바라보는 방향
     - victim_point가 없으면 robot5 pose orientation 사용
  4. Nav 성공 시에만 /robot6/mission/arrived 발행
  5. control 노드가 arrived를 받아 세션을 시작하고, /robot6/tts/done(True) 수신 시 다음 목표 진행
  6. 큐가 비면 pre-dock 위치로 이동 후 navigator.dock() 수행

주의:
  - 현재 nav의 정식 입력은 /rescue/victim_pose_stamped 가 아니라 robot5 토픽 체인이다.
  - INIT_POSE_* 는 실맵 기준으로 반드시 수정해야 합니다.
  - pre-dock 실패 시 dock action은 시도하지 않습니다.
    정렬 실패 상태에서 바로 dock 하면 실기기에서 더 위험할 수 있습니다.
"""

import math
import threading
import time
from collections import deque

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Twist
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TaskResult


ROBOT5_POSE_TOPIC = '/robot5/robot_pose_at_detection'
VICTIM_POINT_TOPIC = '/robot5/victim_point'
VICTIM_POINT_FRESHNESS_SEC = 2.0

ARRIVED_TOPIC = '/robot6/mission/arrived'
MISSION_TIMEOUT_TOPIC = '/robot6/mission/timeout'
TTS_DONE_TOPIC = '/robot6/tts/done'
STOP_TOPIC = '/robot/stop'

TF_STABLE_WAIT_SEC = 2.0
QUEUE_WAIT_SEC = 0.3
MISSION_WAIT_TIMEOUT = 120.0

UNDOCK_FORWARD_VEL = 0.15
UNDOCK_FORWARD_DIST = 0.7

INIT_POSE_X = -0.742
INIT_POSE_Y = -1.71
INIT_POSE_YAW = 0.344

DOCK_POSE_X = -1.4139
DOCK_POSE_Y = -1.8058
DOCK_POSE_YAW = 0.1736


def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def make_pose_stamped(node: Node, x: float, y: float, yaw: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0
    pose.pose.orientation = yaw_to_quaternion(yaw)
    return pose


class RescueNavNode(Node):
    def __init__(self):
        super().__init__('rescue_nav_node')

        self.goal_queue: deque = deque()
        self.queue_lock = threading.Lock()
        self.queue_event = threading.Event()
        self.mission_event = threading.Event()

        self.nav_active = False
        self.nav_ready = False
        self.tf_ready = False
        self._shutdown = False
        self.navigator = None
        self.stop_requested = False

        self.current_goal = None
        self.latest_victim_point = None
        self.latest_victim_point_stamp_sec = None
        self.latest_victim_point_received_t = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            PoseStamped,
            ROBOT5_POSE_TOPIC,
            self.robot_pose_at_detection_callback,
            10,
        )
        self.create_subscription(
            PointStamped,
            VICTIM_POINT_TOPIC,
            self.victim_point_callback,
            10,
        )
        self.create_subscription(Bool, TTS_DONE_TOPIC, self.mission_finished_callback, 10)
        self.create_subscription(Bool, STOP_TOPIC, self.stop_callback, 10)

        self.arrived_pub = self.create_publisher(Bool, ARRIVED_TOPIC, 10)
        self.timeout_pub = self.create_publisher(Bool, MISSION_TIMEOUT_TOPIC, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot6/cmd_vel', 10)

        self.get_logger().info(f'[Init] TF 안정화 대기 중... ({TF_STABLE_WAIT_SEC}초 후 시작)')
        self._start_timer = self.create_timer(TF_STABLE_WAIT_SEC, self._on_tf_ready)

        self._worker = threading.Thread(
            target=self._goal_worker,
            daemon=True,
            name='GoalWorker',
        )
        self._worker.start()

        self.create_timer(3.0, self._status_log)
        self.get_logger().info('[Init] 노드 초기화 완료. Navigator 준비 대기 중...')

    def init_navigator(self):
        self.get_logger().info('[Init] Navigator 초기화 시작...')
        self.navigator = TurtleBot4Navigator(namespace='/robot6')

        nav2_already_active = False
        try:
            import threading as _threading

            ready_evt = _threading.Event()

            def _wait():
                try:
                    self.navigator.waitUntilNav2Active()
                    ready_evt.set()
                except Exception:
                    pass

            t = _threading.Thread(target=_wait, daemon=True)
            t.start()
            nav2_already_active = ready_evt.wait(timeout=5.0)
        except Exception:
            nav2_already_active = False

        if not nav2_already_active:
            self.get_logger().info(
                f'[Init] Nav2 미활성 -> setInitialPose 적용 '
                f'x={INIT_POSE_X:.3f}, y={INIT_POSE_Y:.3f}, yaw={INIT_POSE_YAW:.3f}'
            )
            initial_pose = self._build_nav_pose(INIT_POSE_X, INIT_POSE_Y, INIT_POSE_YAW)
            self.navigator.setInitialPose(initial_pose)
            self.navigator.waitUntilNav2Active()
        else:
            self.get_logger().info('[Init] Nav2 이미 활성 상태 -> setInitialPose 생략 (AMCL 유지)')

        self.nav_ready = True
        self.get_logger().info(f'[Init] Nav2 활성화 완료. 현재 도킹 상태: {self._is_docked()}')

    def _build_nav_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        return make_pose_stamped(self, x, y, yaw)

    def _is_docked(self) -> bool:
        if self.navigator is None:
            return True
        try:
            return bool(self.navigator.getDockedStatus())
        except Exception:
            return True

    def _on_tf_ready(self):
        self.tf_ready = True
        self.get_logger().info('[TF] TF 안정화 완료. 요구조자 토픽 대기 중...')
        self._start_timer.cancel()

    def mission_finished_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info('[Mission] 미션 완료 신호 수신됨.')
            self.mission_event.set()

    def stop_callback(self, msg: Bool):
        if not msg.data:
            return
        if self.stop_requested:
            self.get_logger().warn('[Stop] 이미 정지 처리 중입니다.')
            return

        self.get_logger().warn('[Stop] /robot/stop 수신 -> 즉시 정지 후 도킹 복귀 시작!')
        self.stop_requested = True

        try:
            if self.navigator is not None:
                self.navigator.cancelTask()
        except Exception as exc:
            self.get_logger().warn(f'[Stop] cancelTask 실패: {exc}')

        with self.queue_lock:
            self.goal_queue.clear()
            self.get_logger().info('[Stop] goal_queue 초기화 완료.')

        self.mission_event.set()
        self.queue_event.set()

    def victim_point_callback(self, msg: PointStamped):
        point = self._transform_point_to_map(
            x=msg.point.x,
            y=msg.point.y,
            frame=msg.header.frame_id,
            stamp=msg.header.stamp,
        )
        if point is None:
            self.get_logger().warn('[Victim] victim_point를 map으로 변환하지 못해 캐시하지 않습니다.')
            return

        self.latest_victim_point = point
        self.latest_victim_point_stamp_sec = self._stamp_to_sec(msg.header.stamp)
        self.latest_victim_point_received_t = time.time()
        self.get_logger().info(
            f'[Victim] victim_point 캐시 갱신 -> map({point[0]:.3f}, {point[1]:.3f})'
        )

    def robot_pose_at_detection_callback(self, msg: PoseStamped):
        fallback_yaw = quaternion_to_yaw(msg.pose.orientation)
        self.get_logger().info(
            f'[Robot5] robot_pose_at_detection 수신 -> frame={msg.header.frame_id}, '
            f'x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, '
            f'fallback_yaw={fallback_yaw:.3f}'
        )
        self._enqueue_robot5_pose(
            x=msg.pose.position.x,
            y=msg.pose.position.y,
            fallback_yaw=fallback_yaw,
            frame=msg.header.frame_id,
            stamp=msg.header.stamp,
        )

    def _resolve_goal_yaw(
        self,
        goal_x: float,
        goal_y: float,
        fallback_yaw: float,
        pose_stamp=None,
    ) -> float:
        victim = self.latest_victim_point
        if victim is None:
            self.get_logger().warn('[Yaw] victim_point 캐시 없음 -> robot5 pose orientation 사용')
            return fallback_yaw

        pose_stamp_sec = self._stamp_to_sec(pose_stamp)
        victim_stamp_sec = self.latest_victim_point_stamp_sec
        if pose_stamp_sec is not None and victim_stamp_sec is not None:
            if abs(victim_stamp_sec - pose_stamp_sec) > VICTIM_POINT_FRESHNESS_SEC:
                self.get_logger().warn('[Yaw] victim_point stamp 차이 큼 -> robot5 pose orientation 사용')
                return fallback_yaw
        elif time.time() - self.latest_victim_point_received_t > VICTIM_POINT_FRESHNESS_SEC:
            self.get_logger().warn('[Yaw] victim_point 캐시 stale -> robot5 pose orientation 사용')
            return fallback_yaw

        dx = victim[0] - goal_x
        dy = victim[1] - goal_y
        if math.hypot(dx, dy) < 1e-4:
            self.get_logger().warn('[Yaw] goal과 victim_point가 거의 동일 -> robot5 pose orientation 사용')
            return fallback_yaw
        goal_yaw = math.atan2(dy, dx)
        self.get_logger().info(
            f'[Yaw] victim_point 기준 yaw 계산 -> victim=({victim[0]:.3f}, {victim[1]:.3f}), yaw={goal_yaw:.3f}'
        )
        return goal_yaw

    def _transform_point_to_map(self, x: float, y: float, frame: str, stamp=None):
        if not self.tf_ready:
            self.get_logger().warn('[TF] TF 미준비. 해당 좌표 변환 생략.')
            return None

        if frame == 'map' or frame == '':
            return (x, y)

        use_stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        pt = PointStamped()
        pt.header.frame_id = frame
        pt.point.x, pt.point.y, pt.point.z = x, y, 0.0

        for use_latest_time in (False, True):
            try:
                if use_latest_time:
                    from rclpy.time import Time as RclpyTime

                    pt.header.stamp = RclpyTime().to_msg()
                else:
                    pt.header.stamp = use_stamp
                pt_map = self.tf_buffer.transform(pt, 'map', timeout=Duration(seconds=1.0))
                mode = 'Time(0) 폴백' if use_latest_time else '원본 stamp'
                self.get_logger().info(
                    f'[TF] {frame} -> map 변환 성공 ({mode}): ({pt_map.point.x:.3f}, {pt_map.point.y:.3f})'
                )
                return (pt_map.point.x, pt_map.point.y)
            except Exception as exc:
                if not use_latest_time:
                    self.get_logger().warn(
                        f'[TF] {frame} -> map 변환 실패 (원본 stamp): {exc}. Time(0) 폴백 시도...'
                    )
                else:
                    self.get_logger().error(
                        f'[TF] {frame} -> map 변환 최종 실패: {exc} -> 해당 좌표 버림'
                    )
        return None

    def _enqueue_robot5_pose(self, x: float, y: float, fallback_yaw: float, frame: str, stamp=None):
        point = self._transform_point_to_map(x=x, y=y, frame=frame, stamp=stamp)
        if point is None:
            self.get_logger().warn('[Queue] robot5 pose를 map으로 변환하지 못해 목표를 버립니다.')
            return

        map_x, map_y = point
        goal_yaw = self._resolve_goal_yaw(
            goal_x=map_x,
            goal_y=map_y,
            fallback_yaw=fallback_yaw,
            pose_stamp=stamp,
        )

        with self.queue_lock:
            self.goal_queue.append((map_x, map_y, goal_yaw))
            q_len = len(self.goal_queue)

        self.get_logger().info(
            f'[Queue] + robot5 검출 위치 목표 추가 -> map({map_x:.3f}, {map_y:.3f}), yaw={goal_yaw:.3f} | '
            f'큐 길이: {q_len}'
        )
        self.queue_event.set()

    def _stamp_to_sec(self, stamp) -> float | None:
        if stamp is None:
            return None
        sec = getattr(stamp, 'sec', 0)
        nanosec = getattr(stamp, 'nanosec', 0)
        if sec == 0 and nanosec == 0:
            return None
        return float(sec) + float(nanosec) / 1_000_000_000.0

    def _goal_worker(self):
        self.get_logger().info('[Worker] 목표 워커 시작.')

        while not self._shutdown and not self.nav_ready:
            time.sleep(0.5)
        if self._shutdown:
            return

        self.get_logger().info('[Worker] Navigator 준비 완료. 목표 대기 시작.')

        while not self._shutdown and rclpy.ok():
            if self.stop_requested:
                self.get_logger().warn('[Worker] stop_requested 감지 -> pre-dock 후 도킹 복귀.')
                self.nav_active = False
                self.current_goal = None
                self._go_predock_and_dock()
                self.stop_requested = False
                self.get_logger().info('[Worker] 도킹 완료. 정지 대기...')
                self.queue_event.wait()
                self.queue_event.clear()
                continue

            goal = self._pop_next_goal()

            if goal is None:
                self._go_predock_and_dock()
                self.get_logger().info('[Worker] 큐 비어있음 -> 새 목표 대기...')
                self.queue_event.wait()
                self.queue_event.clear()
                continue

            map_x, map_y, goal_yaw = goal

            self._do_undock()

            self.nav_active = True
            self.current_goal = (map_x, map_y)
            goal_pose = self._build_nav_pose(map_x, map_y, goal_yaw)

            self.get_logger().info(
                f'[Worker] 주행 시작 -> map({map_x:.3f}, {map_y:.3f}), yaw={goal_yaw:.3f}'
            )

            success, result = self._go_to_pose_blocking(goal_pose)

            if success:
                self._publish_arrived(True)
                self._wait_for_mission_completion()
            else:
                self.get_logger().warn(f'[Worker] 목표 실패. arrived 미발행. result={result}')

            self.nav_active = False
            self.current_goal = None

        self.get_logger().info('[Worker] 워커 종료.')

    def _pop_next_goal(self):
        with self.queue_lock:
            if self.goal_queue:
                return self.goal_queue.popleft()
        return None

    def _go_to_pose_blocking(self, pose: PoseStamped):
        try:
            self.navigator.goToPose(pose)
            while rclpy.ok() and not self._shutdown:
                if self.stop_requested:
                    self.get_logger().warn('[Nav] stop_requested -> 주행 즉시 중단.')
                    break
                if self.navigator.isTaskComplete():
                    break
                time.sleep(QUEUE_WAIT_SEC)

            result = self.navigator.getResult()
            self.get_logger().info(
                f'[Nav] 완료 -> x={pose.pose.position.x:.3f}, '
                f'y={pose.pose.position.y:.3f}, result={result}'
            )
            return self._is_nav_success(result), result
        except Exception as exc:
            self.get_logger().error(f'[Nav] goToPose 실패: {exc}')
            return False, str(exc)

    def _is_nav_success(self, result) -> bool:
        if result is None:
            return False
        try:
            if result == TaskResult.SUCCEEDED:
                return True
        except Exception:
            pass
        name = getattr(result, 'name', None)
        if isinstance(name, str):
            return name.upper() in ('SUCCEEDED', 'SUCCESS')
        text = str(result).lower()
        if 'succeed' in text or 'success' in text:
            return True
        if 'fail' in text or 'cancel' in text or 'unknown' in text:
            return False
        value = getattr(result, 'value', None)
        if isinstance(value, int):
            return value == 0
        return False

    def _publish_arrived(self, arrived: bool):
        msg = Bool()
        msg.data = arrived
        self.arrived_pub.publish(msg)
        self.get_logger().info(f'[Worker] arrived={arrived} 발행')

    def _wait_for_mission_completion(self):
        self.get_logger().info(f'[Worker] 구조 미션 대기 중... ({TTS_DONE_TOPIC})')
        self.mission_event.clear()
        finished = self.mission_event.wait(timeout=MISSION_WAIT_TIMEOUT)
        if finished:
            self.get_logger().info('[Worker] TTS/구조 완료 신호 수신. 다음 동작으로 이동.')
        else:
            self.get_logger().warn(
                f'[Worker] 미션 대기 시간 초과({MISSION_WAIT_TIMEOUT}초). '
                f'{MISSION_TIMEOUT_TOPIC} 발행 후 강제 진행.'
            )
            msg = Bool()
            msg.data = True
            self.timeout_pub.publish(msg)
            self.get_logger().info(f'[Worker] timeout=True 발행 ({MISSION_TIMEOUT_TOPIC})')

    def _go_predock_and_dock(self):
        if self._is_docked():
            return

        predock_pose = self._compute_predock_pose()
        self.get_logger().info(
            f'[Dock] pre-dock 이동 시작 -> '
            f'x={predock_pose.pose.position.x:.3f}, y={predock_pose.pose.position.y:.3f}'
        )

        success, result = self._go_to_pose_blocking(predock_pose)
        if not success:
            self.get_logger().error(
                f'[Dock] pre-dock 이동 실패. direct dock 생략. result={result}'
            )
            return

        self.get_logger().info('[Dock] pre-dock 정렬 완료. dock 실행.')
        self._do_dock()

    def _compute_predock_pose(self) -> PoseStamped:
        return self._build_nav_pose(DOCK_POSE_X, DOCK_POSE_Y, DOCK_POSE_YAW)

    def _do_dock(self):
        if self._is_docked():
            return
        try:
            self.get_logger().info('[Dock] 도킹 시작...')
            self.navigator.dock()
            self.get_logger().info(f'[Dock] 도킹 호출 완료. 현재 상태={self._is_docked()}')
        except Exception as exc:
            self.get_logger().error(f'[Dock] 도킹 실패: {exc}')

    def _do_undock(self):
        if not self._is_docked():
            return
        try:
            self.get_logger().info('[Dock] 새 목표 수신 -> 언도킹 시작...')
            self.navigator.undock()
            self.get_logger().info(f'[Dock] 언도킹 호출 완료. 현재 상태={self._is_docked()}')
            self._forward_after_undock()
        except Exception as exc:
            self.get_logger().error(f'[Dock] 언도킹 실패: {exc}')

    def _forward_after_undock(self):
        dist = UNDOCK_FORWARD_DIST
        vel = UNDOCK_FORWARD_VEL
        duration = dist / vel
        dt = 0.05

        self.get_logger().info(
            f'[Undock] 전진 시작 -> {dist:.1f}m @ {vel:.2f}m/s '
            f'(예상 {duration:.1f}초)'
        )

        twist_fwd = Twist()
        twist_fwd.linear.x = vel
        twist_stop = Twist()

        elapsed = 0.0
        while elapsed < duration and not self._shutdown and rclpy.ok():
            self.cmd_vel_pub.publish(twist_fwd)
            time.sleep(dt)
            elapsed += dt

        self.cmd_vel_pub.publish(twist_stop)
        self.get_logger().info(f'[Undock] 전진 완료 (실제 {elapsed:.2f}초 이동)')

    def _status_log(self):
        with self.queue_lock:
            q_len = len(self.goal_queue)
        self.get_logger().info(
            f'[Status] 도킹={self._is_docked()} | 주행중={self.nav_active} | '
            f'현재목표={self.current_goal} | 대기목표={q_len}개'
        )

    def shutdown(self):
        self.get_logger().info('[Shutdown] 종료 시작...')
        self._shutdown = True
        self.queue_event.set()

        try:
            if self.navigator is not None:
                self.navigator.cancelTask()
        except Exception:
            pass

        try:
            if self.navigator is not None and not self._is_docked():
                self.get_logger().info('[Shutdown] pre-dock 후 도킹 복귀 시도...')
                self._go_predock_and_dock()
        except Exception:
            pass

        self.get_logger().info('[Shutdown] 완료.')


def main():
    rclpy.init()
    node = RescueNavNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    init_thread = threading.Thread(target=node.init_navigator, daemon=True)
    init_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C 감지 -> 종료')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# person_event_detector (preview raw image version)
# - RGB:    /robot5/oakd/rgb/preview/image_raw
# - K:      /robot5/oakd/rgb/preview/camera_info
# - Depth:  /robot5/oakd/stereo/image_raw
# - 평소엔 화면 표시
# - scan_active == True 일 때만 YOLO 추론
# - 사람 검출 시
#     1) victim_point
#     2) victim_event_json
#     3) robot_pose_at_detection
#   발행
# -----------------------------------------------------------------------------

import json
import time
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point  # noqa: F401
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class PersonEventDetector(Node):
    def __init__(self):
        super().__init__('person_event_detector')

        # ===== params =====
        self.weights = 'yolo11n.pt'
        self.target_class_id = 0
        self.conf_thres = 0.25
        self.infer_period = 0.20

        self.depth_min = 0.20
        self.depth_max = 5.00
        self.depth_patch_r = 3
        self.depth_offset = -0.04

        self.event_dup_dist = 0.8
        self.event_cooldown = 10.0
        self.recent_events = []

        self.view = True
        self.base_frames = ['base_link', 'base_footprint']

        # ===== internal =====
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.K = None
        self.camera_frame = None

        self.latest_rgb_bgr = None
        self.latest_rgb_stamp = None

        self.latest_depth_m = None
        self.latest_depth_stamp = None
        self.latest_depth_frame = None

        self.display_image = None
        self.scan_active = False
        self.last_infer_t = 0.0
        self.gui_initialized = False

        ns = self.get_namespace().rstrip('/')

        # preview 사용
        self.rgb_topic = f'{ns}/oakd/rgb/preview/image_raw' if ns else '/oakd/rgb/preview/image_raw'
        self.info_topic = f'{ns}/oakd/rgb/preview/camera_info' if ns else '/oakd/rgb/preview/camera_info'

        # depth는 기존 stereo 유지
        self.depth_topic = f'{ns}/oakd/stereo/image_raw' if ns else '/oakd/stereo/image_raw'

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.model = YOLO(self.weights)
        self.get_logger().info(f'YOLO loaded: {self.weights}')

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(CameraInfo, self.info_topic, self.camera_info_cb, sensor_qos)
        self.create_subscription(Image, self.rgb_topic, self.rgb_cb, sensor_qos)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, sensor_qos)
        self.create_subscription(Bool, 'scan_active', self.scan_active_cb, 10)

        self.victim_point_pub = self.create_publisher(PointStamped, 'victim_point', 10)
        self.victim_event_pub = self.create_publisher(String, 'victim_event_json', 10)
        self.robot_pose_pub = self.create_publisher(PoseStamped, 'robot_pose_at_detection', 10)
        self.detector_image_pub = self.create_publisher(Image, 'detector_yolo', 10)

        self.timer = self.create_timer(0.05, self.step)
        self.gui_timer = self.create_timer(0.03, self.gui_step)

        self.get_logger().info('person_event_detector started')

    # =========================
    # callbacks
    # =========================
    def camera_info_cb(self, msg: CameraInfo):
        with self.lock:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.camera_frame = msg.header.frame_id

    def rgb_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if bgr is None:
                return

            with self.lock:
                self.latest_rgb_bgr = bgr
                self.latest_rgb_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f'rgb cb error: {e}')

    def depth_cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_m = self.depth_to_m(depth)

            with self.lock:
                self.latest_depth_m = depth_m
                self.latest_depth_stamp = msg.header.stamp
                self.latest_depth_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f'depth cb error: {e}')

    def scan_active_cb(self, msg: Bool):
        self.scan_active = bool(msg.data)

    # =========================
    # helper
    # =========================
    def depth_to_m(self, depth_np: np.ndarray) -> np.ndarray:
        if depth_np.dtype == np.uint16:
            depth_f = depth_np.astype(np.float32) / 1000.0
        else:
            depth_f = depth_np.astype(np.float32)

        valid = depth_f > 0.0
        depth_f[valid] += self.depth_offset
        depth_f[depth_f < 0.0] = 0.0
        return depth_f

    def get_latest_inputs(self):
        with self.lock:
            rgb = None if self.latest_rgb_bgr is None else self.latest_rgb_bgr.copy()
            depth_m = None if self.latest_depth_m is None else self.latest_depth_m.copy()
            K = None if self.K is None else self.K.copy()
            camera_frame = self.camera_frame

        if rgb is None or depth_m is None or K is None or camera_frame is None:
            return None

        return {
            'rgb': rgb,
            'depth_m': depth_m,
            'K': K,
            'frame_id': camera_frame,
        }

    def sample_depth(self, depth_m: np.ndarray, x: int, y: int):
        h, w = depth_m.shape[:2]
        x = clamp(int(x), 0, w - 1)
        y = clamp(int(y), 0, h - 1)

        z = float(depth_m[y, x])

        if (not np.isfinite(z)) or z <= 0.01:
            r = self.depth_patch_r
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            patch = depth_m[y0:y1, x0:x1]
            patch = patch[np.isfinite(patch)]
            patch = patch[patch > 0.01]
            if patch.size == 0:
                return None
            z = float(np.median(patch))

        if not (self.depth_min <= z <= self.depth_max):
            return None

        return z

    def pick_person_bbox(self, yolo_res):
        boxes = yolo_res.boxes
        if boxes is None or len(boxes) == 0:
            return None

        best = None
        best_score = -1.0

        for b in boxes:
            cls_id = int(b.cls.item())
            if cls_id != self.target_class_id:
                continue

            conf = float(b.conf.item())
            if conf < self.conf_thres:
                continue

            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            area = max(1.0, (x2 - x1) * (y2 - y1))
            score = conf * np.sqrt(area)

            if score > best_score:
                best_score = score
                best = (conf, (x1, y1, x2, y2))

        return best

    def bbox_to_map_point(self, bbox, rgb_shape, depth_m, K, frame_id):
        x1, y1, x2, y2 = bbox

        # preview 영상 기준 사람 몸통 쪽 픽셀
        u_rgb = int((x1 + x2) * 0.5)
        v_rgb = int(y1 + (y2 - y1) * 0.55)

        rgb_h, rgb_w = rgb_shape[:2]
        depth_h, depth_w = depth_m.shape[:2]

        u_rgb = clamp(u_rgb, 0, rgb_w - 1)
        v_rgb = clamp(v_rgb, 0, rgb_h - 1)

        # preview -> depth 해상도 스케일
        sx = float(depth_w) / float(rgb_w)
        sy = float(depth_h) / float(rgb_h)

        u_depth = int(round(u_rgb * sx))
        v_depth = int(round(v_rgb * sy))

        u_depth = clamp(u_depth, 0, depth_w - 1)
        v_depth = clamp(v_depth, 0, depth_h - 1)

        z = self.sample_depth(depth_m, u_depth, v_depth)
        if z is None:
            return None

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # 3D는 preview camera_info 기준으로 계산
        X = (u_rgb - cx) * z / fx
        Y = (v_rgb - cy) * z / fy
        Z = z

        pt_camera = PointStamped()
        pt_camera.header.stamp = Time().to_msg()
        pt_camera.header.frame_id = frame_id
        pt_camera.point.x = float(X)
        pt_camera.point.y = float(Y)
        pt_camera.point.z = float(Z)

        try:
            pt_map = self.tf_buffer.transform(pt_camera, 'map', timeout=Duration(seconds=0.5))
            return {
                'map_x': float(pt_map.point.x),
                'map_y': float(pt_map.point.y),
                'map_z': float(pt_map.point.z),
                'pix_x': int(u_rgb),
                'pix_y': int(v_rgb),
                'depth_pix_x': int(u_depth),
                'depth_pix_y': int(v_depth),
                'depth_m': float(z),
            }
        except Exception as e:
            self.get_logger().warn(f'tf transform failed: {e}')
            return None

    def get_robot_pose_in_map(self):
        for base_frame in self.base_frames:
            try:
                tf_robot = self.tf_buffer.lookup_transform(
                    'map',
                    base_frame,
                    Time(),
                    timeout=Duration(seconds=0.5),
                )

                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'map'
                pose_msg.pose.position.x = tf_robot.transform.translation.x
                pose_msg.pose.position.y = tf_robot.transform.translation.y
                pose_msg.pose.position.z = tf_robot.transform.translation.z
                pose_msg.pose.orientation = tf_robot.transform.rotation
                return pose_msg
            except Exception:
                continue

        return None

    def prune_recent_events(self, now_t: float):
        self.recent_events = [
            ev for ev in self.recent_events
            if now_t - ev['t'] <= self.event_cooldown
        ]

    def is_duplicate_event(self, x: float, y: float, now_t: float):
        self.prune_recent_events(now_t)
        for ev in self.recent_events:
            if np.hypot(x - ev['x'], y - ev['y']) <= self.event_dup_dist:
                return True
        return False

    def publish_event(self, candidate, conf):
        now = time.time()
        x = float(candidate['map_x'])
        y = float(candidate['map_y'])
        z = float(candidate['map_z'])

        if self.is_duplicate_event(x, y, now):
            return

        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = 'map'
        pt.point.x = x
        pt.point.y = y
        pt.point.z = z
        self.victim_point_pub.publish(pt)

        robot_pose = self.get_robot_pose_in_map()
        if robot_pose is not None:
            self.robot_pose_pub.publish(robot_pose)
            self.get_logger().info(
                f'robot_pose_at_detection: '
                f'({robot_pose.pose.position.x:.2f}, {robot_pose.pose.position.y:.2f}), '
                f'yaw={quat_to_yaw(robot_pose.pose.orientation):.2f}'
            )
        else:
            self.get_logger().warn('robot pose at detection lookup failed')

        payload = {
            'stamp': now,
            'robot_id': self.get_namespace().strip('/') or 'robot5',
            'map_x': x,
            'map_y': y,
            'map_z': z,
            'confidence': float(conf),
            'depth_m': float(candidate['depth_m']),
            'target_class': 'person',
            'depth_pix_x': int(candidate['depth_pix_x']),
            'depth_pix_y': int(candidate['depth_pix_y']),
        }

        if robot_pose is not None:
            payload.update({
                'robot_map_x': float(robot_pose.pose.position.x),
                'robot_map_y': float(robot_pose.pose.position.y),
                'robot_map_z': float(robot_pose.pose.position.z),
                'robot_qx': float(robot_pose.pose.orientation.x),
                'robot_qy': float(robot_pose.pose.orientation.y),
                'robot_qz': float(robot_pose.pose.orientation.z),
                'robot_qw': float(robot_pose.pose.orientation.w),
                'robot_yaw': float(quat_to_yaw(robot_pose.pose.orientation)),
            })

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.victim_event_pub.publish(msg)

        self.recent_events.append({'x': x, 'y': y, 't': now})

        self.get_logger().info(
            f'victim_event published: ({x:.2f}, {y:.2f}, {z:.2f}) conf={conf:.2f}'
        )

    def render_overlay(self, frame, picked=None, candidate=None):
        img = frame.copy()

        if picked is not None:
            conf, (x1, y1, x2, y2) = picked
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f'person {conf:.2f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        if candidate is not None:
            cv2.circle(
                img,
                (candidate['pix_x'], candidate['pix_y']),
                4,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                img,
                f'z={candidate["depth_m"]:.2f}m',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        status_text = 'SCAN_ACTIVE' if self.scan_active else 'IDLE'
        status_color = (0, 255, 255) if self.scan_active else (180, 180, 180)
        cv2.putText(
            img,
            status_text,
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
        )

        with self.lock:
            self.display_image = img

        try:
            msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.camera_frame if self.camera_frame is not None else ''
            self.detector_image_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'detector_yolo publish error: {e}')

    # =========================
    # main loop
    # =========================
    def step(self):
        with self.lock:
            latest_rgb = None if self.latest_rgb_bgr is None else self.latest_rgb_bgr.copy()

        if latest_rgb is not None:
            self.render_overlay(latest_rgb, None, None)

        if not self.scan_active:
            return

        now = time.time()
        if now - self.last_infer_t < self.infer_period:
            return

        inputs = self.get_latest_inputs()
        if inputs is None:
            return

        self.last_infer_t = now

        rgb = inputs['rgb']
        depth_m = inputs['depth_m']
        K = inputs['K']
        frame_id = inputs['frame_id']

        try:
            res = self.model.predict(
                rgb,
                conf=self.conf_thres,
                classes=[self.target_class_id],
                verbose=False,
            )[0]
        except Exception as e:
            self.get_logger().warn(f'YOLO predict error: {e}')
            return

        picked = self.pick_person_bbox(res)
        candidate = None

        if picked is not None:
            conf, bbox = picked
            candidate = self.bbox_to_map_point(bbox, rgb.shape, depth_m, K, frame_id)
            if candidate is not None:
                self.publish_event(candidate, conf)

            self.render_overlay(rgb, picked, candidate)
        else:
            self.render_overlay(rgb, None, None)

    # =========================
    # GUI
    # =========================
    def gui_step(self):
        if not self.view:
            return

        with self.lock:
            img = None if self.display_image is None else self.display_image.copy()

        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                img,
                'Waiting for RGB...',
                (140, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

        if not self.gui_initialized:
            cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('RGB', 720, 720)
            self.gui_initialized = True

        cv2.imshow('RGB', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PersonEventDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
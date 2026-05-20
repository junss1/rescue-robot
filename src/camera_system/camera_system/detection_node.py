"""
Combined Detection Node
- 두 카메라로부터 압축 이미지 입력 받음
- Person 감지 (YOLO segmentation)
- Turtlebot 감지 (custom YOLO)
- 타임스탐프 기반 동기화로 overlay_node에 전달
"""

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, String


@dataclass
class YoloConfig:
    """YOLO 추론 설정"""
    imgsz: int = 960          # 실제 추론 시 사용할 이미지 크기
    warmup_imgsz: int = 640   # 워밍업 시 사용할 이미지 크기


@dataclass
class PersonModelConfig:
    """Person 감지 모델 설정"""
    model_file: str = 'yolov8n-seg.pt'
    label: str = 'person'
    conf_threshold: float = 0.15      # 신뢰도 임계값 (낮을수록 민감)
    iou_threshold: float = 0.45       # IoU 임계값 (낮을수록 더 많이 탐지)
    min_contour_area: int = 100       # 최소 contour 크기 (노이즈 필터링)


@dataclass
class TurtlebotModelConfig:
    """Turtlebot 감지 모델 설정"""
    model_file: str = 'best26.pt'
    label: str = 'turtlebot'
    conf_threshold: float = 0.25      # 신뢰도 임계값
    iou_threshold: float = 0.90       # IoU 임계값 (높을수록 필터링 강함)
    smoothing_buffer_size: int = 5    # Moving average를 위한 버퍼 크기


CAMERAS = ('cam01', 'cam02')
YOLO_CONFIG = YoloConfig()
PERSON_CONFIG = PersonModelConfig()
TURTLEBOT_CONFIG = TurtlebotModelConfig()


class ModelInference:
    """
    YOLO 모델 추론을 담당하는 클래스
    
    기능:
    - 모델 로드 및 워밍업
    - 프레임에 대한 추론 실행
    - GPU/CPU 자동 선택
    - Half-precision (FP16) 지원
    """

    def __init__(self, model_path: str, device: str, use_half: bool, logger):
        self.model = YOLO(model_path)
        self.device = device
        self.use_half = use_half
        self.logger = logger

    def warmup(self, imgsz: int):
        """
        모델 워밍업
        
        목적: 첫 추론 속도 저하 해결 (메모리 할당 등)
        
        인자:
        - imgsz: 워밍업 이미지 크기
        """
        warmup = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        try:
            self.model(
                warmup,
                verbose=False,
                conf=0.5,
                iou=0.5,
                imgsz=imgsz,
                device=self.device,
                half=self.use_half,
            )
        except Exception as exc:
            self.logger.warning(f'모델 워밍업 실패: {exc}')

    def infer(self, frame, conf: float, iou: float, retina_masks: bool = False):
        """
        추론 실행
        
        인자:
        - frame: 입력 프레임 (numpy array, BGR)
        - conf: 신뢰도 임계값
        - iou: IoU (비최대 억제) 임계값
        - retina_masks: 마스크 품질 (True = 고품질이지만 느림)
        
        반환:
        - YOLO 결과 객체 목록
        """
        return self.model(
            frame,
            verbose=False,
            conf=conf,
            iou=iou,
            retina_masks=retina_masks,
            imgsz=YOLO_CONFIG.imgsz,
            device=self.device,
            half=self.use_half,
        )


class CombinedDetectNode(Node):
    """
    통합 객체 감지 노드
    
    기능:
    - 두 개의 USB 카메라(cam01, cam02)로부터 압축 이미지 수신
    - Person 감지 (YOLO segmentation)
    - Turtlebot 감지 (Custom YOLO)
    - 타임스탐프 기반 동기화로 결과 발행
    - 부드러운 Turtlebot 개수 추정 (moving average)
    
    토픽:
    - 입력: /camera/{cam}/raw (CompressedImage)
    - 출력: /detection/{cam}/person, /detection/{cam}/turtlebot (JSON String)
    - 출력: /detection/summary/person, /detection/summary/turtlebot (Int32)
    """

    def __init__(self):
        super().__init__('detect_node')

        # QoS 설정 (신뢰할 수 있는 전송)
        image_qos = QoSProfile(depth=1)
        image_qos.reliability = ReliabilityPolicy.RELIABLE
        result_qos = QoSProfile(depth=10)
        result_qos.reliability = ReliabilityPolicy.RELIABLE

        # 카메라별 상태 초기화 (person/turtlebot 개수, 버퍼 등)
        self._init_camera_states()
        
        # 카메라로부터 이미지 & detection 결과 구독
        self._create_subscriptions(image_qos)
        
        # 감지 결과를 발행할 퍼블리셔 생성
        self._create_publishers(result_qos)
        
        # GPU/CPU 설정 및 YOLO 모델 로드
        self._setup_device()
        self._load_models()

        # 메인 처리 루프 타이머 (10ms = 100Hz, 카메라 10FPS 대비 충분)
        self.worker_timer = self.create_timer(0.01, self._process_latest_frame)
        
        # 모든 모델 워밍업 (첫 추론 속도 저하 제거)
        self._warmup_all_models()
        self.get_logger().info('Combined detect node started.')

    def _init_camera_states(self) -> None:
        """카메라별 상태 변수 초기화"""
        self.latest_person_count = {camera: 0 for camera in CAMERAS}
        self.latest_turtlebot_count = {camera: 0 for camera in CAMERAS}
        self.count_buffer = {
            camera: deque(maxlen=TURTLEBOT_CONFIG.smoothing_buffer_size)
            for camera in CAMERAS
        }
        self.last_turtlebot_detections = {camera: [] for camera in CAMERAS}
        self.pending_msg = {camera: None for camera in CAMERAS}
        self.processing = {camera: False for camera in CAMERAS}

    def _create_subscriptions(self, qos: QoSProfile) -> None:
        """모든 카메라의 이미지 구독자 생성"""
        for camera in CAMERAS:
            self.create_subscription(
                CompressedImage,
                f'/camera/{camera}/raw',
                lambda msg, cam=camera: self._image_callback(msg, cam),
                qos,
            )

    def _create_publishers(self, qos: QoSProfile) -> None:
        """Person, Turtlebot, 합계 퍼블리셔 생성"""
        self.person_pub = {
            camera: self.create_publisher(
                String,
                f'/detection/{camera}/person',
                qos,
            )
            for camera in CAMERAS
        }

        self.turtlebot_pub = {
            camera: self.create_publisher(
                String,
                f'/detection/{camera}/turtlebot',
                qos,
            )
            for camera in CAMERAS
        }

        self.person_total_pub = self.create_publisher(
            Int32,
            '/detection/summary/person',
            qos,
        )
        self.turtlebot_total_pub = self.create_publisher(
            Int32,
            '/detection/summary/turtlebot',
            qos,
        )

    def _setup_device(self) -> None:
        """
        GPU/CPU 설정 확인
        
        - CUDA 가용성 확인
        - Half-precision (FP16) 지원 여부 판별
        - 사용할 디바이스 선택
        """
        self.use_cuda = torch.cuda.is_available()
        self.yolo_device = 'cuda:0' if self.use_cuda else 'cpu'
        self.use_half = self.use_cuda

        if self.use_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            self.get_logger().info(
                f'CUDA enabled | device={self.yolo_device} | gpu={gpu_name} | half={self.use_half}'
            )
        else:
            self.get_logger().warning('CUDA 미지원, CPU 추론 사용')

    def _load_models(self) -> None:
        """
        YOLO 모델 로드
        
        모델:
        - Person: yolov8n-seg.pt (segmentation)
        - Turtlebot: best26.pt (custom detection)
        """
        person_model_path = self._find_model_path(PERSON_CONFIG.model_file)
        turtlebot_model_path = self._find_model_path(TURTLEBOT_CONFIG.model_file)

        self.get_logger().info(f'Person YOLO 모델 로드: {person_model_path}')
        self.person_inference = ModelInference(
            person_model_path,
            self.yolo_device,
            self.use_half,
            self.get_logger(),
        )

        self.get_logger().info(f'Turtlebot YOLO 모델 로드: {turtlebot_model_path}')
        self.turtlebot_inference = ModelInference(
            turtlebot_model_path,
            self.yolo_device,
            self.use_half,
            self.get_logger(),
        )

    def _find_model_path(self, model_file: str) -> str:
        """모델 파일 경로 찾기 (ROS2 패키지 또는 로컬)"""
        try:
            # ROS2 패키지 공유 디렉토리에서 찾기
            pkg_share_dir = get_package_share_directory('camera_system')
            return os.path.join(pkg_share_dir, 'models', model_file)
        except Exception:
            # 개발 중: 상위 디렉토리의 models 폴더에서 찾기
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            return os.path.join(base_dir, 'models', model_file)

    def _warmup_all_models(self):
        """모든 모델을 워밍업한다."""
        self.person_inference.warmup(YOLO_CONFIG.warmup_imgsz)
        self.turtlebot_inference.warmup(YOLO_CONFIG.warmup_imgsz)
        self.get_logger().info('YOLO warmup complete.')

    def _image_callback(self, msg: CompressedImage, camera_name: str):
        """카메라별 최신 메시지만 유지한다."""
        self.pending_msg[camera_name] = msg

    def _process_latest_frame(self):
        """
        각 카메라의 최신 프레임을 person / turtlebot 순서로 처리
        
        동작:
        1. 각 카메라의 최신 메시지 확인
        2. CompressedImage → BGR 프레임으로 디코딩
        3. Person 감지 (segmentation masks 추출)
        4. Turtlebot 감지 (bounding boxes)
        5. **동일 타임스탐프로** 결과를 JSON으로 발행
        """
        for camera_name in CAMERAS:
            if self.processing[camera_name]:
                continue

            msg = self.pending_msg[camera_name]
            if msg is None:
                continue

            self.pending_msg[camera_name] = None
            self.processing[camera_name] = True

            try:
                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    self.get_logger().warning(f'[{camera_name}] 프레임 디코딩 실패')
                    self._publish_empty_person(camera_name, msg)
                    self._publish_empty_turtlebot(camera_name, msg)
                    continue

                # ⭐ 동일 타임스탐프로 두 감지 실행 (overlay_node 동기화 용)

                self._process_person(camera_name, frame, msg)
                self._process_turtlebot(camera_name, frame, msg)
            finally:
                self.processing[camera_name] = False

    @staticmethod
    def _stamp_tuple(msg: CompressedImage) -> tuple:
        """메시지 타임스탬프 추출"""
        return int(msg.header.stamp.sec), int(msg.header.stamp.nanosec)

    def _make_payload(self, camera_name: str, msg: CompressedImage, detections):
        """JSON payload 생성"""
        stamp_sec, stamp_nanosec = self._stamp_tuple(msg)
        return {
            'camera_name': camera_name,
            'stamp_sec': stamp_sec,
            'stamp_nanosec': stamp_nanosec,
            'detections': detections,
        }

    def _publish_string(self, publisher, payload):
        """JSON 문자열 발행"""
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(msg)

    def _publish_int(self, publisher, value: int):
        """정수 발행"""
        msg = Int32()
        msg.data = int(value)
        publisher.publish(msg)

    def _publish_empty_person(self, camera_name: str, msg: CompressedImage):
        """person 검출 실패 시 빈 결과 발행"""
        self._publish_string(
            self.person_pub[camera_name],
            self._make_payload(camera_name, msg, {
                'person_count': 0,
                'boxes': [],
                'contours': [],
            }),
        )
        self.latest_person_count[camera_name] = 0
        self._publish_int(self.person_total_pub, sum(self.latest_person_count.values()))

    def _publish_empty_turtlebot(self, camera_name: str, msg: CompressedImage):
        """turtlebot 검출 실패 시 빈 결과 발행"""
        self.count_buffer[camera_name].append(0)
        buf = self.count_buffer[camera_name]
        smoothed_count = int(round(sum(buf) / len(buf))) if buf else 0

        self._publish_string(
            self.turtlebot_pub[camera_name],
            self._make_payload(camera_name, msg, {
                'turtlebot_count': smoothed_count,
                'boxes': [],
            }),
        )
        self.latest_turtlebot_count[camera_name] = smoothed_count
        self._publish_int(self.turtlebot_total_pub, sum(self.latest_turtlebot_count.values()))

    def _process_person(self, camera_name: str, frame, msg: CompressedImage):
        """
        Person 감지 실행 및 결과 발행
        
        동작:
        1. YOLO segmentation 모델로 추론
        2. 각 감지된 사람마다 bounding box 및 contour 추출
        3. 결과를 JSON으로 패킹하여 발행
        
        발행 토픽:
        - /detection/{camera}/person: {person_count, boxes, contours}
        - /detection/summary/person: 전체 사람 수
        """
        try:
            results = self.person_inference.infer(
                frame,
                conf=PERSON_CONFIG.conf_threshold,
                iou=PERSON_CONFIG.iou_threshold,
                retina_masks=True,
            )
        except Exception as exc:
            self.get_logger().error(f'[{camera_name}] person 모델 실행 오류: {exc}')
            self._publish_empty_person(camera_name, msg)
            return

        stamp_sec, stamp_nanosec = self._stamp_tuple(msg)
        box_detections = []
        contour_detections = []
        person_count = 0

        for result in results:
            boxes = result.boxes
            masks = result.masks
            if boxes is None:
                continue

            mask_data = None
            if masks is not None and masks.data is not None:
                mask_data = masks.data.cpu().numpy()

            for index, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = self.person_inference.model.names.get(cls_id, str(cls_id))

                if class_name != PERSON_CONFIG.label:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                contour_points = self._extract_contour(mask_data, index)

                box_detections.append({
                    'class_name': class_name,
                    'conf': conf,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                })
                contour_detections.append({
                    'class_name': class_name,
                    'conf': conf,
                    'contour': contour_points,
                })
                person_count += 1

        # 발행 (통합 JSON: count + boxes + contours)
        payload = self._make_payload(camera_name, msg, {
            'person_count': person_count,
            'boxes': box_detections,
            'contours': contour_detections,
        })
        self._publish_string(self.person_pub[camera_name], payload)
        
        # 디버깅 로그
        # if person_count > 0 or (len(self.latest_person_count) > 0):
        #     self.get_logger().info(
        #         f'[{camera_name}] PERSON: count={person_count}, boxes={len(box_detections)}, contours={len(contour_detections)}'
        #     )
        
        self.latest_person_count[camera_name] = person_count
        self._publish_int(self.person_total_pub, sum(self.latest_person_count.values()))

    @staticmethod
    def _extract_contour(mask_data, index: int) -> list:
        """
        분할 마스크에서 contour를 추출
        
        방법:
        1. mask_data에서 해당 인덱스의 마스크 추출
        2. OpenCV findContours로 경계선 찾기
        3. 가장 큰 contour만 반환
        
        반환:
        - [[x1, y1], [x2, y2], ...] 형태의 좌표 리스트
        """
        if mask_data is None or index >= len(mask_data):
            return []

        mask = mask_data[index]
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > PERSON_CONFIG.min_contour_area:
            return largest.squeeze(1).tolist()

        return []

    def _process_turtlebot(self, camera_name: str, frame, msg: CompressedImage):
        """
        Turtlebot 감지 실행 및 결과 발행
        
        특징:
        - Moving average로 개수를 부드럽게 처리
        - 연속 프레임의 개수를 버퍼에 저장하고 평균 계산
        
        발행 토픽:
        - /detection/{camera}/turtlebot: {turtlebot_count (smoothed), boxes}
        - /detection/summary/turtlebot: 전체 터틀봇 수 (smoothed)
        """
        try:
            results = self.turtlebot_inference.infer(
                frame,
                conf=TURTLEBOT_CONFIG.conf_threshold,
                iou=TURTLEBOT_CONFIG.iou_threshold,
            )
        except Exception as exc:
            self.get_logger().error(f'[{camera_name}] turtlebot 모델 실행 오류: {exc}')
            self._publish_empty_turtlebot(camera_name, msg)
            return

        stamp_sec, stamp_nanosec = self._stamp_tuple(msg)
        current_detections = []
        raw_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = self.turtlebot_inference.model.names.get(cls_id, str(cls_id))

                if class_name != TURTLEBOT_CONFIG.label:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                current_detections.append({
                    'class_name': class_name,
                    'conf': conf,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                })
                raw_count += 1

        # Smoothing count
        self.count_buffer[camera_name].append(raw_count)
        buf = self.count_buffer[camera_name]
        smoothed_count = int(round(sum(buf) / len(buf)))

        # 발행 (통합 JSON: count + boxes)
        payload = self._make_payload(camera_name, msg, {
            'turtlebot_count': smoothed_count,
            'boxes': current_detections,
        })
        self._publish_string(self.turtlebot_pub[camera_name], payload)
        
        # # 디버깅 로그
        # if smoothed_count > 0 or raw_count > 0:
        #     self.get_logger().info(
        #         f'[{camera_name}] TURTLEBOT: raw={raw_count}, smoothed={smoothed_count}, boxes={len(current_detections)}'
        #     )
        
        self.latest_turtlebot_count[camera_name] = smoothed_count
        self._publish_int(self.turtlebot_total_pub, sum(self.latest_turtlebot_count.values()))


def main(args=None):
    """
    메인 진입점
    
    ROS2 노드 시작:
    1. rclpy 초기화
    2. CombinedDetectNode 생성 및 스핀
    3. Ctrl+C로 종료
    """
    rclpy.init(args=args)
    node = None
    try:
        node = CombinedDetectNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('[INFO] Interrupted by user')
    except Exception as e:
        print(f'[ERROR] {e}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

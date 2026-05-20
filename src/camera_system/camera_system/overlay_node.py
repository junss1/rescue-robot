# overlay_node.py v0.003 2026-03-16
# [이번 버전에서 수정된 사항]
# - 붕괴 경고 문구를 우측 상단으로 이동하고 크기를 줄여 화면을 덜 가리도록 조정
"""
Overlay Node
- 원본 이미지와 detection 결과를 합쳐서 시각화
- 타임스탐프 기반 버퍼 관리로 동기화
"""

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Bool


@dataclass
class BufferConfig:
    """타임스탐프 기반 버퍼 설정"""
    max_size: int = 30                  # 최대 버퍼 크기 (너무 크면 메모리 낭비)
    max_age_ns: int = 1_000_000_000     # 최대 보존 시간 (1초)
    cleanup_period_sec: float = 0.5     # 주기적 정리 간격


@dataclass
class VisualizationConfig:
    """이미지 시각화 설정 (색상, 폰트, 두께 등)"""
    # 객체별 색상1 (BGR 형식)
    person_color: Tuple[int, int, int] = (0, 255, 0)       # 초록색
    turtlebot_color: Tuple[int, int, int] = (0, 0, 255)    # 빨간색
    text_color: Tuple[int, int, int] = (255, 255, 255)     # 흰색
    alert_border_color: Tuple[int, int, int] = (0, 0, 255) # 빨간색 (경고)
    alert_overlay_color: Tuple[int, int, int] = (0, 0, 255) # 빨간색 오버레이
    
    # 폰트 설정
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6            # 기본 폰트 크기
    font_thickness: int = 2            # 폰트 두께
    
    # 그리기 설정
    box_thickness: int = 2             # Bounding box 선 두께
    contour_thickness: int = 2         # Contour 선 두께
    alert_border_thickness: int = 8    # Alert 경고 테두리 두께
    alert_overlay_alpha: float = 0.2   # Alert 오버레이 투명도
    jpeg_quality: int = 80             # JPEG 압축 품질


@dataclass
class TopicConfig:
    """ROS2 토픽 이름 설정"""
    image_format: str = '/camera/{}/raw'
    overlay_format: str = '/output/{}/compressed'
    person_contours_format: str = '/detection/{}/person'
    turtlebot_boxes_format: str = '/detection/{}/turtlebot'
    alert_format: str = '/alert/{}/collapse'
    diff_image_format: str = '/alert/{}/diff'


CAMERAS = ('cam01', 'cam02')
BUFFER_CONFIG = BufferConfig()
VISUALIZATION_CONFIG = VisualizationConfig()
TOPIC_CONFIG = TopicConfig()


class DetectionBuffer:
    """
    카메라별 detection 동기화 버퍼
    
    목적:
    - 이미지, Person 감지, Turtlebot 감지가 같은 타임스탐프로 도착하기를 대기
    - 타임스탐프 기반 매칭으로 동기화
    - 오래된 데이터 자동 정리
    """

    def __init__(self, max_size: int, max_age_ns: int):
        self.max_size = max_size
        self.max_age_ns = max_age_ns
        self.image_buffer: OrderedDict[int, CompressedImage] = OrderedDict()
        self.person_buffer: OrderedDict[int, List] = OrderedDict()
        self.turtlebot_buffer: OrderedDict[int, List] = OrderedDict()
        self.latest_key = 0

    def add_image(self, key: int, msg: CompressedImage):
        """이미지 버퍼에 추가"""
        self.latest_key = max(self.latest_key, key)
        self.image_buffer[key] = msg

    def add_person(self, key: int, detections: List):
        """Person 감지 결과 추가"""
        self.person_buffer[key] = detections

    def add_turtlebot(self, key: int, detections: List):
        """Turtlebot 감지 결과 추가"""
        self.turtlebot_buffer[key] = detections

    def get_all(self, key: int) -> Optional[Tuple]:
        """
        동일한 타임스탐프의 모든 데이터를 반환하고 제거
        
        반환:
        - (image, person_detections, turtlebot_detections) 튜플
        - 데이터가 불완전하면 None
        """
        if not (key in self.image_buffer and key in self.person_buffer and key in self.turtlebot_buffer):
            return None

        image = self.image_buffer.pop(key)
        person = self.person_buffer.pop(key)
        turtlebot = self.turtlebot_buffer.pop(key)
        return image, person, turtlebot

    def has_all(self, key: int) -> bool:
        """모든 데이터가 준비되었는지 확인"""
        return (
            key in self.image_buffer
            and key in self.person_buffer
            and key in self.turtlebot_buffer
        )

    def cleanup_old(self, expire_before: int):
        """지정된 시간 이전의 오래된 데이터를 삭제 (타이밍 아웃)"""
        for buffer_dict in (self.image_buffer, self.person_buffer, self.turtlebot_buffer):
            stale_keys = [key for key in buffer_dict if key < expire_before]
            for stale_key in stale_keys:
                buffer_dict.pop(stale_key, None)

    def trim(self):
        """버퍼 크기 제한"""
        for buffer_dict in (self.image_buffer, self.person_buffer, self.turtlebot_buffer):
            while len(buffer_dict) > self.max_size:
                buffer_dict.popitem(last=False)


class OverlayNode(Node):
    """
    Overlay 시각화 노드
    
    기능:
    - 원본 이미지와 detection 결과를 합쳐서 시각화
    - 타임스탐프 기반 버퍼로 멀티카메라 동기화
    - Person contours 및 Turtlebot boxes 그리기
    - Collapse alert 시각화 (경고 테두리 및 차분 이미지 표시)
    
    입력 토픽:
    - /camera/{cam}/raw: 원본 이미지
    - /detection/{cam}/person: Person 감지 결과
    - /detection/{cam}/turtlebot: Turtlebot 감지 결과
    - /alert/{cam}/collapse: 붕괴 경보
    - /alert/{cam}/diff: 차분 이미지
    
    출력 토픽:
    - /output/{cam}: Overlay된 시각화 이미지 (CompressedImage)
    """

    def __init__(self):
        super().__init__('overlay_node')

        # QoS 설정
        image_qos = QoSProfile(depth=1)
        image_qos.reliability = ReliabilityPolicy.RELIABLE
        result_qos = QoSProfile(depth=10)
        result_qos.reliability = ReliabilityPolicy.RELIABLE

        # 카메라별 버퍼 초기화
        self.buffers = {
            camera: DetectionBuffer(BUFFER_CONFIG.max_size, BUFFER_CONFIG.max_age_ns)
            for camera in CAMERAS
        }

        # 구독자 생성
        self._create_subscriptions(image_qos, result_qos)

        # 퍼블리셔 생성 (CompressedImage)
        self.overlay_pub = {
            camera: self.create_publisher(
                CompressedImage,
                TOPIC_CONFIG.overlay_format.format(camera),
                image_qos,
            )
            for camera in CAMERAS
        }

        # 최신 상태 캐시
        self.latest_person_contours = {camera: [] for camera in CAMERAS}
        self.latest_turtlebot_boxes = {camera: [] for camera in CAMERAS}
        self.latest_alerts = {camera: {'collapse_detected': False} for camera in CAMERAS}
        self.latest_diff_images = {camera: None for camera in CAMERAS}

        # 주기 타이머
        self.cleanup_timer = self.create_timer(
            BUFFER_CONFIG.cleanup_period_sec,
            self._cleanup_old_buffers,
        )

        self.get_logger().info('Overlay node started.')

    def _create_subscriptions(self, image_qos: QoSProfile, result_qos: QoSProfile):
        """모든 구독자를 생성한다"""
        for camera in CAMERAS:
            # 이미지 구독
            self.create_subscription(
                CompressedImage,
                TOPIC_CONFIG.image_format.format(camera),
                lambda msg, cam=camera: self._image_callback(msg, cam),
                image_qos,
            )

            # Person contours 구독
            self.create_subscription(
                String,
                TOPIC_CONFIG.person_contours_format.format(camera),
                lambda msg, cam=camera: self._person_contours_callback(msg, cam),
                result_qos,
            )

            # Turtlebot boxes 구독
            self.create_subscription(
                String,
                TOPIC_CONFIG.turtlebot_boxes_format.format(camera),
                lambda msg, cam=camera: self._turtlebot_boxes_callback(msg, cam),
                result_qos,
            )

            # Collapse alert 구독
            self.create_subscription(
                Bool,
                TOPIC_CONFIG.alert_format.format(camera),
                lambda msg, cam=camera: self._alert_callback(msg, cam),
                result_qos,
            )

            # Collapse diff_image 구독
            self.create_subscription(
                CompressedImage,
                TOPIC_CONFIG.diff_image_format.format(camera),
                lambda msg, cam=camera: self._diff_image_callback(msg, cam),
                image_qos,
            )

    @staticmethod
    def _stamp_to_key(sec: int, nanosec: int) -> int:
        """타임스탬프를 정수 키로 변환"""
        return int(sec) * 1_000_000_000 + int(nanosec)

    def _parse_payload(self, msg: String) -> Tuple[Optional[int], dict]:
        """JSON payload 파싱 (새 형식: detections은 dict)"""
        try:
            data = json.loads(msg.data)
        except Exception:
            return None, {}

        if isinstance(data, dict):
            sec = data.get('stamp_sec')
            nanosec = data.get('stamp_nanosec')
            detections = data.get('detections', {})
            if sec is None or nanosec is None:
                return None, detections if isinstance(detections, dict) else {}
            return (
                self._stamp_to_key(sec, nanosec),
                detections if isinstance(detections, dict) else {},
            )

        return None, {}

    def _person_contours_callback(self, msg: String, camera_name: str):
        """Person contours 콜백 (새 형식: dict 전체 저장)"""
        key, detection_data = self._parse_payload(msg)
        if key is None:
            # 타임스탐프가 없으면 버퍼에 저장하지 않고 반환 (None을 key로 저장하면 publish 실패)
            self.latest_person_contours[camera_name] = []
            return

        # 새 형식: detection_data = {person_count: ..., boxes: [...], contours: [...]}
        # 전체 dict를 버퍼에 저장 (나중에 _create_overlay에서 파싱)
        self.buffers[camera_name].add_person(key, detection_data)
        contours = detection_data.get('contours', []) if isinstance(detection_data, dict) else []
        self.latest_person_contours[camera_name] = contours
        self._try_publish(camera_name, key)
        self._trim_all_buffers()

    def _turtlebot_boxes_callback(self, msg: String, camera_name: str):
        """Turtlebot boxes 콜백 (새 형식: dict 전체 저장)"""
        key, detection_data = self._parse_payload(msg)
        if key is None:
            # 타임스탐프가 없으면 버퍼에 저장하지 않고 반환 (None을 key로 저장하면 publish 실패)
            self.latest_turtlebot_boxes[camera_name] = []
            return

        # 새 형식: detection_data = {turtlebot_count: ..., boxes: [...]}
        # 전체 dict를 버퍼에 저장 (나중에 _create_overlay에서 파싱)
        self.buffers[camera_name].add_turtlebot(key, detection_data)
        boxes = detection_data.get('boxes', []) if isinstance(detection_data, dict) else []
        self.latest_turtlebot_boxes[camera_name] = boxes
        self._try_publish(camera_name, key)
        self._trim_all_buffers()

    def _alert_callback(self, msg: Bool, camera_name: str):
        """Collapse alert 콜백"""
        try:
            # Bool 메시지 직접 사용
            self.latest_alerts[camera_name] = {'collapse_detected': msg.data}
            
            # Alert가 해제되면 diff_image도 초기화
            if not msg.data:
                self.latest_diff_images[camera_name] = None
        except Exception:
            self.latest_alerts[camera_name] = {'collapse_detected': False}
            self.latest_diff_images[camera_name] = None

    def _diff_image_callback(self, msg: CompressedImage, camera_name: str):
        """Collapse diff_image 콜백"""
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.latest_diff_images[camera_name] = frame
        except Exception:
            self.latest_diff_images[camera_name] = None

    def _image_callback(self, msg: CompressedImage, camera_name: str):
        """이미지 콜백"""
        key = self._stamp_to_key(msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.buffers[camera_name].add_image(key, msg)
        self._try_publish(camera_name, key)
        self._trim_all_buffers()

    def _try_publish(self, camera_name: str, key: int):
        """동일한 타임스탐프의 모든 데이터가 준비되면 overlay를 발행한다"""
        buffer = self.buffers[camera_name]
        if not buffer.has_all(key):
            return

        result = buffer.get_all(key)
        if result is None:
            return

        image_msg, person_detections, turtlebot_detections = result
        self.latest_person_contours[camera_name] = person_detections
        self.latest_turtlebot_boxes[camera_name] = turtlebot_detections

        # person_detections과 turtlebot_detections은 dict (전체 detection data)
        overlay_msg = self._create_overlay(
            image_msg,
            person_detections,  # dict
            turtlebot_detections,  # dict
            camera_name,
            self.latest_alerts[camera_name],
            self.latest_diff_images[camera_name],
        )
        if overlay_msg is not None:
            # CompressedImage 발행
            self.overlay_pub[camera_name].publish(overlay_msg)

    def _create_overlay(
        self,
        image_msg: CompressedImage,
        person_detections: Dict,
        turtlebot_detections: Dict,
        camera_name: str,
        alert_data: Dict,
        diff_image: Optional[np.ndarray],
    ) -> Optional[CompressedImage]:
        """
        Overlay 이미지 생성
        
        동작:
        1. CompressedImage를 BGR 프레임으로 디코딩
        2. Person contours 그리기 (녹색 다각형)
        3. Turtlebot boxes 그리기 (빨강 사각형)
        4. Alert 중이면 경고 시각화 추가
        5. 카메라 정보 텍스트 추가
        6. JPEG 재인코딩하여 반환
        """
        try:
            np_arr = np.frombuffer(image_msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warning(f'[{camera_name}] 프레임 디코딩 실패')
                return None

            annotated = frame.copy()

            # Person contours 그리기
            person_count = 0
            contours_list = []
            if isinstance(person_detections, dict) and person_detections:  # dict 확인
                person_count = person_detections.get('person_count', 0)
                contours_list = person_detections.get('contours', []) or []
            for det in contours_list:
                self._draw_contour(annotated, det)

            # Turtlebot boxes 그리기
            turtlebot_count = 0
            boxes_list = []
            if isinstance(turtlebot_detections, dict) and turtlebot_detections:  # dict 확인
                turtlebot_count = turtlebot_detections.get('turtlebot_count', 0)
                boxes_list = turtlebot_detections.get('boxes', []) or []
            for det in boxes_list:
                self._draw_box(annotated, det)

            # Alert 중이면 시각화
            if alert_data.get('collapse_detected', False):
                self._draw_alert_overlay(annotated, camera_name, diff_image)

            # 카메라 정보 및 카운트 표시
            self._draw_info_text(
                annotated,
                camera_name,
                person_count,
                turtlebot_count,
            )

            # JPEG 인코딩
            ok, buffer = cv2.imencode(
                '.jpg',
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), VISUALIZATION_CONFIG.jpeg_quality],
            )
            if not ok:
                self.get_logger().warning(f'[{camera_name}] JPEG 인코딩 실패')
                return None

            out_msg = CompressedImage()
            out_msg.header = image_msg.header
            out_msg.format = 'jpeg'
            out_msg.data = buffer.tobytes()
            return out_msg

        except Exception as exc:
            self.get_logger().error(f'[{camera_name}] Overlay 생성 오류: {exc}')
            return None

    def _draw_contour(
        self,
        image: np.ndarray,
        det: Dict,
    ):
        """
        Person contour를 이미지에 그리기
        
        시각화:
        - 초록색 다각형으로 사람 윤곽선 표시
        - 좌상단에 클래스명 및 신뢰도 표시
        """
        try:
            contour_points = det.get('contour', [])
            if not contour_points or len(contour_points) < 3:
                return

            contour = np.array(contour_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                image,
                [contour],
                isClosed=True,
                color=VISUALIZATION_CONFIG.person_color,
                thickness=VISUALIZATION_CONFIG.contour_thickness,
            )

            x, y, _, _ = cv2.boundingRect(contour)
            class_name = det.get('class_name', 'person')
            conf = float(det.get('conf', 0.0))
            self._draw_label(image, f'{class_name} {conf:.2f}', x, y, VISUALIZATION_CONFIG.person_color)
        except Exception:
            pass

    def _draw_box(
        self,
        image: np.ndarray,
        det: Dict,
    ):
        """
        Turtlebot bounding box를 이미지에 그리기
        
        시각화:
        - 빨강 사각형으로 터틀봇 범위 표시
        - 좌상단에 클래스명 및 신뢰도 표시
        """
        try:
            x1 = int(det.get('x1', 0))
            y1 = int(det.get('y1', 0))
            x2 = int(det.get('x2', 0))
            y2 = int(det.get('y2', 0))

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                VISUALIZATION_CONFIG.turtlebot_color,
                VISUALIZATION_CONFIG.box_thickness,
            )

            class_name = det.get('class_name', 'turtlebot')
            conf = float(det.get('conf', 0.0))
            self._draw_label(image, f'{class_name} {conf:.2f}', x1, y1, VISUALIZATION_CONFIG.turtlebot_color)
        except Exception:
            pass

    @staticmethod
    def _draw_label(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ):
        """라벨 텍스트를 이미지에 표시"""
        cv2.putText(
            image,
            text,
            (x, max(y - 10, 20)),
            VISUALIZATION_CONFIG.font,
            VISUALIZATION_CONFIG.font_scale,
            color,
            VISUALIZATION_CONFIG.font_thickness,
        )

    def _draw_alert_overlay(self, image: np.ndarray, camera_name: str, diff_image: Optional[np.ndarray]):
        """
        Alert 상태를 이미지에 시각화
        
        표시 사항:
        - 빨강 경고 테두리 (전체 프레임)
        - 붕괴 영역 빨강 오버레이 (차분 이미지 기반)
        - 상단 중앙에 경고 텍스트
        """
        try:
            h, w = image.shape[:2]

            # 차분 이미지가 있으면 붕괴 영역만 강조
            if diff_image is not None:
                try:
                    # 차분 이미지를 현재 프레임 크기에 맞춤
                    if diff_image.shape[:2] != (h, w):
                        diff_img_resized = cv2.resize(diff_image, (w, h))
                    else:
                        diff_img_resized = diff_image

                    # Grayscale이면 마스크로 사용
                    if len(diff_img_resized.shape) == 2:
                        diff_mask = diff_img_resized > 50
                    else:
                        # BGR이면 회색을 마스크로 변환
                        diff_gray = cv2.cvtColor(diff_img_resized, cv2.COLOR_BGR2GRAY)
                        diff_mask = diff_gray > 50

                    # 붕괴 영역(마스크가 True인 부분)만 빨강으로 오버레이
                    if np.any(diff_mask):
                        overlay = image.copy()
                        overlay[diff_mask] = VISUALIZATION_CONFIG.alert_overlay_color
                        
                        # 차분 영역만 반투명 합성
                        image[diff_mask] = cv2.addWeighted(
                            overlay[diff_mask],
                            0.5,  # 차분 영역 투명도
                            image[diff_mask],
                            0.5,
                            0,
                        )
                except Exception as exc:
                    self.get_logger().debug(f'[{camera_name}] 차분 이미지 합성 오류: {exc}')

            # 빨강 테두리 (경계 강조)
            cv2.rectangle(
                image,
                (0, 0),
                (w - 1, h - 1),
                VISUALIZATION_CONFIG.alert_border_color,
                VISUALIZATION_CONFIG.alert_border_thickness,
            )

            # 이모티콘 없이 영문 경고 문구를 우측 상단에 표시한다.
            alert_text = 'COLLAPSE DETECTED'
            text_scale = 0.75
            text_margin = 20
            text_size, _ = cv2.getTextSize(
                alert_text,
                VISUALIZATION_CONFIG.font,
                text_scale,
                VISUALIZATION_CONFIG.font_thickness + 1,
            )
            cv2.putText(
                image,
                alert_text,
                (max(text_margin, w - text_size[0] - text_margin), 40),
                VISUALIZATION_CONFIG.font,
                text_scale,
                VISUALIZATION_CONFIG.alert_border_color,
                VISUALIZATION_CONFIG.font_thickness + 1,
            )
        except Exception as exc:
            self.get_logger().error(f'[{camera_name}] Alert overlay 오류: {exc}')

    def _draw_info_text(
        self,
        image: np.ndarray,
        camera_name: str,
        person_count: int,
        turtlebot_count: int,
    ):
        """
        카메라 정보 및 감지 카운트를 표시
        
        표시 항목:
        - 카메라 이름
        - Person 개수 (초록색)
        - Turtlebot 개수 (빨강색)
        """
        texts = [
            (f'Camera: {camera_name}', 40),
            (f'Person: {person_count}', 80),
            (f'Turtlebot: {turtlebot_count}', 120),
        ]

        for text, y in texts:
            if 'Person' in text:
                color = VISUALIZATION_CONFIG.person_color
            elif 'Turtlebot' in text:
                color = VISUALIZATION_CONFIG.turtlebot_color
            else:
                color = VISUALIZATION_CONFIG.text_color

            cv2.putText(
                image,
                text,
                (20, y),
                VISUALIZATION_CONFIG.font,
                1.0 if 'Camera' in text else VISUALIZATION_CONFIG.font_scale,
                color,
                VISUALIZATION_CONFIG.font_thickness,
            )

    def _cleanup_old_buffers(self):
        """
        오래된 버퍼 항목 정기적 정리
        
        목적:
        - 도착하지 않은 데이터 제거 (타임아웃)
        - 메모리 누수 방지
        """
        for camera_name in CAMERAS:
            buffer = self.buffers[camera_name]
            if buffer.latest_key <= 0:
                continue

            expire_before = max(0, buffer.latest_key - BUFFER_CONFIG.max_age_ns)
            buffer.cleanup_old(expire_before)

        self._trim_all_buffers()

    def _trim_all_buffers(self):
        """모든 버퍼의 크기를 최대값 이내로 제한"""
        for camera_name in CAMERAS:
            self.buffers[camera_name].trim()

    def destroy_node(self):
        """
        노드 정리 및 종료
        
        정리 항목:
        - 정기 타이머 취소
        - OpenCV 윈도우 닫기
        - ROS2 노드 파괴
        """
        try:
            if hasattr(self, 'cleanup_timer'):
                self.cleanup_timer.cancel()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        return super().destroy_node()


def main(args=None):
    """
    메인 진입점
    
    ROS2 노드 시작:
    1. rclpy 초기화
    2. OverlayNode 생성 및 스핀
    3. Ctrl+C로 종료
    """
    rclpy.init(args=args)
    node = None
    try:
        node = OverlayNode()
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

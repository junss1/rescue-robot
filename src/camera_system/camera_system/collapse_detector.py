"""
Background Collapse Detection Node
- 두 카메라로부터 원본 이미지와 detection 결과 받음
- 사람/터틀봇 영역 제외 후 구조물 ROI에서만 차분 계산
- 연속 프레임 조건으로 붕괴 징후 감지
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool


@dataclass
class CollapseDetectorConfig:
    """붕괴 감지 알고리즘 설정"""
    diff_threshold: int = 20            # 픽셀 변화 임계값 (0-255)
    area_ratio_threshold: float = 0.03  # 변화 영역이 차지하는 비율 임계값 (3%)
    min_frames_to_alert: int = 10       # 경보 발생에 필요한 연속 프레임 (1초 @ 10Hz)
    buffer_size: int = 30               # 변화 버퍼 크기 (최근 프레임들 추적)
    require_dual_camera: bool = False   # 듀얼 카메라 조건 (미사용)
    require_dual_frames: int = 2        # 필요한 카메라 다중성 (미사용)
    blur_kernel: int = 5                # 가우시안 블러 커널 크기
    morphology_kernel: int = 5          # 형태학 연산 커널 크기
    jpeg_quality: int = 80              # 차분 이미지 JPEG 품질


@dataclass
class CameraROI:
    """
    카메라별 관심 영역(ROI) 설정
    
    ROI: 구조물(배경)만 모니터링 할 영역
    제외 구역: 사람/로봇이 자주 움직이는 영역 (동적 마스크로 처리)
    """
    camera_name: str
    structure_roi: Tuple[int, int, int, int] = (0, 0, 800, 600)  # (x1, y1, x2, y2) 또는 비율
    exclude_zones: List[Tuple[int, int, int, int]] = None            # 추가 제외 영역
    
    def __post_init__(self):
        if self.exclude_zones is None:
            self.exclude_zones = []


# ========== 카메라별 ROI 및 감지 설정 ==========
# 형식 설명: ROI는 비율 기반 (0.0~1.0)으로 설정하면 실제 프레임 크기에 자동 적응

CAMERA_ROIS = {
    'cam01': CameraROI(camera_name='cam01', structure_roi=(0.0, 0.0, 1.0, 1.0)),  # 전체 영역 모니터링
    'cam02': CameraROI(camera_name='cam02', structure_roi=(0.0, 0.0, 1.0, 1.0)),  # 전체 영역 모니터링
}

# 카메라별 감지 민감도 설정
# cam01: 가까운 거리에서 세밀한 감지
# cam02: 먼 거리에서의 감지 (낮은 임계값 적용)
CAMERA_DETECTOR_CONFIG = {
    'cam01': {
        'diff_threshold': 20,          # 픽셀 변화 임계값
        'area_ratio_threshold': 0.03,  # 3% 이상 변화 감지
        'min_frames_to_alert': 10,     # 약 1초 (10Hz * 10 frames)
    },
    'cam02': {
        'diff_threshold': 15,          # 더 낮은 임계값 (민감함)
        'area_ratio_threshold': 0.02,  # 2% 이상 변화 감지 (더 민감)
        'min_frames_to_alert': 8,      # 더 빨리 감지 (약 0.8초)
    },
}

DETECTOR_CONFIG = CollapseDetectorConfig()
CAMERAS = ('cam01', 'cam02')


class DynamicObjectMask:
    """
    동적 객체 마스크 생성 클래스
    
    목적:
    - Person contours와 Turtlebot boxes로부터 마스크 생성
    - 차분 계산 시 이들 영역을 배제 (사람/로봇 움직임 != 붕괴)
    """
    
    @staticmethod
    def create_mask_from_contours(
        frame_h: int,
        frame_w: int,
        person_contours: List,
        dilate_pixels: int = 50,
    ) -> np.ndarray:
        """
        Person contours로부터 마스크 생성
        
        변경 사항:
        - contour를 채운 후 dilate_pixels만큼 확장하여
          세그먼트 경계 오차 및 미세 움직임으로 인한 오탐지 방지
        
        반환:
        - uint8 마스크 (255: 사람 영역 + 여유, 0: 배경)
        """
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        
        for detection in person_contours:
            try:
                contour_data = detection.get('contour', [])
                if not contour_data or not isinstance(contour_data, list):
                    continue
                
                # 좌표 배열로 변환
                points = np.array(contour_data, dtype=np.int32)
                if len(points) > 2:
                    # 다각형으로 채우기
                    cv2.fillPoly(mask, [points], 255)
            except Exception:
                continue
        
        # 마스크를 dilate_pixels만큼 확장 (사람 경계 여유)
        if dilate_pixels > 0 and np.any(mask):
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1),
            )
            mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        
        return mask
    
    @staticmethod
    def create_mask_from_boxes(
        frame_h: int,
        frame_w: int,
        turtlebot_boxes: List,
    ) -> np.ndarray:
        """
        Turtlebot bounding boxes로부터 마스크 생성
        
        반환:
        - uint8 마스크 (255: 로봇 영역, 0: 배경)
        """
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        
        for detection in turtlebot_boxes:
            try:
                # Box 형식: {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
                x1 = int(detection.get('x1', 0))
                y1 = int(detection.get('y1', 0))
                x2 = int(detection.get('x2', 0))
                y2 = int(detection.get('y2', 0))
                
                if x1 < x2 and y1 < y2:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            except Exception:
                continue
        
        return mask
    
    @staticmethod
    def create_mask_from_person_boxes(
        frame_h: int,
        frame_w: int,
        person_contours: List,
        padding: int = 30,
    ) -> np.ndarray:
        """
        Person detection의 bounding box로부터 마스크 생성
        
        목적:
        - Contour 마스크보다 넓은 영역을 보수적으로 커버
        - Contour에서 빠지는 가장자리 영역을 bounding box로 보완
        
        반환:
        - uint8 마스크 (255: 사람 bbox 영역 + padding, 0: 배경)
        """
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        
        for detection in person_contours:
            try:
                contour_data = detection.get('contour', [])
                if not contour_data or not isinstance(contour_data, list):
                    continue
                
                points = np.array(contour_data, dtype=np.int32)
                if len(points) < 2:
                    continue
                
                # Contour의 bounding rect 계산 + padding
                x, y, w, h = cv2.boundingRect(points)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame_w, x + w + padding)
                y2 = min(frame_h, y + h + padding)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            except Exception:
                continue
        
        return mask
    
    @staticmethod
    def combine_masks(*masks: np.ndarray) -> np.ndarray:
        """
        여러 마스크를 합쳐서 하나의 동적 객체 마스크 생성
        
        목적:
        - Person 마스크 + Turtlebot 마스크를 OR 연산으로 통합
        - 최종 마스크는 차분 계산에서 제외할 영역
        """
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined = cv2.bitwise_or(combined, mask)
        return combined


class BackgroundChangCalculator:
    """
    배경 변화 계산 클래스
    
    기능:
    - 기준 프레임 초기화 및 업데이트
    - 현재 프레임과의 차분 계산
    - 카메라별 민감도 적용 (diff_threshold, area_ratio_threshold)
    - 형태학 연산으로 노이즈 제거
    """
    
    def __init__(self, config: CollapseDetectorConfig):
        self.default_config = config
        self.camera_configs = CAMERA_DETECTOR_CONFIG  # 카메라별 설정
        self.reference_frame = {}  # camera_name -> frame
    
    def _get_camera_config(self, camera_name: str) -> Dict:
        """카메라별 설정 가져오기"""
        return self.camera_configs.get(camera_name, self.camera_configs['cam01'])
    
    def initialize_reference(self, camera_name: str, frame: np.ndarray):
        """
        기준 프레임 설정 (초기화 또는 리셋 시)
        
        목적:
        - 현재 배경 상태를 기준 프레임으로 설정
        - 이후 모든 차분 계산의 기준점
        """
        config = self._get_camera_config(camera_name)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self.default_config.blur_kernel, self.default_config.blur_kernel),
            0
        )
        self.reference_frame[camera_name] = blurred
    
    def calculate_diff(
        self,
        camera_name: str,
        frame: np.ndarray,
        dynamic_mask: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> Dict:
        """
        프레임과 기준 프레임의 차분 계산
        
        알고리즘:
        1. 현재 프레임을 그레이스케일 + 블러처리
        2. 기준 프레임과의 절대값 차분 계산
        3. 카메라별 임계값 적용
        4. ROI 영역만 추출
        5. 동적 객체 마스크 적용 (사람/로봇 제외)
        6. 형태학 연산 (오픈/클로즈)으로 노이즈 제거
        7. 변화 픽셀 수 및 비율 계산
        
        반환:
        - diff_image: 처리된 차분 이미지
        - change_pixels: 변화 픽셀 수
        - change_ratio: 변화 비율
        - has_change: 유의미한 변화 여부
        """
        if camera_name not in self.reference_frame:
            self.initialize_reference(camera_name, frame)
            return {
                'diff_image': None,
                'change_pixels': 0,
                'change_ratio': 0.0,
                'has_change': False,
            }
        
        config = self._get_camera_config(camera_name)
        diff_threshold = config['diff_threshold']
        area_ratio_threshold = config['area_ratio_threshold']
        
        # 그레이스케일 + 블러
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self.default_config.blur_kernel, self.default_config.blur_kernel),
            0
        )
        
        # 차분 계산
        diff = cv2.absdiff(blurred, self.reference_frame[camera_name])
        
        # 임계값 적용 (카메라별 diff_threshold)
        _, diff_binary = cv2.threshold(
            diff,
            diff_threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        # ROI 적용
        x1, y1, x2, y2 = roi
        roi_region = diff_binary[y1:y2, x1:x2].copy()
        
        # 동적 객체 마스크 적용 (제외)
        if dynamic_mask is not None:
            dynamic_roi = dynamic_mask[y1:y2, x1:x2]
            # 동적 객체 영역 제외
            roi_region = cv2.bitwise_and(
                roi_region,
                cv2.bitwise_not(dynamic_roi)
            )
        
        # 형태학 연산 (노이즈 제거)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.default_config.morphology_kernel, self.default_config.morphology_kernel)
        )
        roi_region = cv2.morphologyEx(roi_region, cv2.MORPH_OPEN, kernel)
        roi_region = cv2.morphologyEx(roi_region, cv2.MORPH_CLOSE, kernel)
        
        # 변화 픽셀 수 계산
        change_pixels = np.count_nonzero(roi_region)
        roi_area = roi_region.size
        change_ratio = change_pixels / roi_area if roi_area > 0 else 0.0
        
        # 유의미한 변화인지 판정 (카메라별 area_ratio_threshold)
        has_change = change_ratio > area_ratio_threshold
        
        return {
            'diff_image': roi_region,
            'change_pixels': change_pixels,
            'change_ratio': change_ratio,
            'has_change': has_change,
        }
    
    def update_reference(self, camera_name: str, frame: np.ndarray):
        """
        기준 프레임 점진적 업데이트 (online learning)
        
        목적:
        - 장시간 실행 시 조명 변화 등 배경 변화에 적응
        - Alpha = 0.05 (느린 업데이트이므로 돌발적 변화는 감지)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self.default_config.blur_kernel, self.default_config.blur_kernel),
            0
        )
        # 느린 업데이트 (alpha=0.05: 새 프레임 5%, 기존 레퍼런스 95%)
        alpha = 0.05
        self.reference_frame[camera_name] = cv2.addWeighted(
            blurred,
            alpha,
            self.reference_frame[camera_name],
            1 - alpha,
            0
        )


class CollapseDetectorNode(Node):
    """
    구조물 붕괴 감지 노드
    
    핵심 기능:
    - 이미지 프레임에서 배경 변화 감지
    - Person/Turtlebot 객체 영역 제외 (동적 마스크)
    - 연속 프레임에서 경보 조건 판정
    - 누적 차분 이미지로 붕괴 부위 시각화
    - 수동 리셋 신호 처리
    
    입력 토픽:
    - /camera/{cam}/raw: 원본 이미지
    - /detection/{cam}/person: Person contours
    - /detection/{cam}/turtlebot: Turtlebot boxes
    - /control/alert/reset: 경보 리셋 신호
    
    출력 토픽:
    - /alert/{cam}/collapse: 붕괴 경보 (True/False)
    - /alert/{cam}/diff: 차분 이미지 (경보 중일 때만)
    """
    
    def __init__(self):
        super().__init__('collapse_detector_node')
        
        # QoS 설정
        image_qos = QoSProfile(depth=1)
        image_qos.reliability = ReliabilityPolicy.RELIABLE
        result_qos = QoSProfile(depth=10)
        result_qos.reliability = ReliabilityPolicy.RELIABLE
        
        # 최신 메시지 저장
        self.latest_image = {camera: None for camera in CAMERAS}
        self.latest_person_contours = {camera: [] for camera in CAMERAS}
        self.latest_turtlebot_boxes = {camera: [] for camera in CAMERAS}
        
        # 카메라별 실제 해상도 (첫 프레임에서 설정됨)
        self.camera_resolutions = {camera: None for camera in CAMERAS}
        
        # 변화 감지 버퍼
        self.change_buffer = {
            camera: deque(maxlen=DETECTOR_CONFIG.buffer_size)
            for camera in CAMERAS
        }
        
        # 이전 Alert 상태 추적
        self.previous_collapse_state = {camera: False for camera in CAMERAS}
        
        # 누적 차분 이미지 (Alert 중에 OR로 누적, 'r' 키로만 초기화)
        self.accumulated_diff_images = {camera: None for camera in CAMERAS}
        
        # 최근 N프레임 person 위치 이력 (이전 위치도 마스킹하기 위함)
        self._person_mask_history_size = 10  # 최근 10프레임 이력 보관
        self._person_mask_history = {
            camera: deque(maxlen=self._person_mask_history_size)
            for camera in CAMERAS
        }
        
        # 차분 계산기
        self.diff_calculator = BackgroundChangCalculator(DETECTOR_CONFIG)
        
        # 구독자 생성
        for camera in CAMERAS:
            # 이미지
            self.create_subscription(
                CompressedImage,
                f'/camera/{camera}/raw',
                lambda msg, cam=camera: self._image_callback(msg, cam),
                image_qos,
            )
            
            # Person contours
            self.create_subscription(
                String,
                f'/detection/{camera}/person',
                lambda msg, cam=camera: self._person_callback(msg, cam),
                result_qos,
            )
            
            # Turtlebot boxes
            self.create_subscription(
                String,
                f'/detection/{camera}/turtlebot',
                lambda msg, cam=camera: self._turtlebot_callback(msg, cam),
                result_qos,
            )
        
        # 퍼블리셔
        self.alert_pub = {
            camera: self.create_publisher(
                Bool,
                f'/alert/{camera}/collapse',
                result_qos,
            )
            for camera in CAMERAS
        }
        
        # 차분 이미지 퍼블리셔
        self.diff_image_pub = {
            camera: self.create_publisher(
                CompressedImage,
                f'/alert/{camera}/diff',
                image_qos,
            )
            for camera in CAMERAS
        }
        
        # Reset 신호 구독
        self.create_subscription(
            String,
            '/control/alert/reset',
            self._reset_residual_callback,
            result_qos,
        )
        
        # 처리 타이머 (10ms = 100Hz, 카메라 10FPS 대비 충분)
        self.timer = self.create_timer(0.01, self._process_frame)
        
        # 파일 기반 reset 신호 체크 타이머 (1초 주기, 파일I/O 부하 최소화)
        self.reset_check_timer = self.create_timer(1.0, self._check_reset_signal)
        
        self.get_logger().info('Collapse detector node started.')
    
    def _image_callback(self, msg: CompressedImage, camera_name: str):
        """
        이미지 콜백
        
        동작:
        - CompressedImage를 BGR 프레임으로 디코딩
        - 최신 이미지만 저장 (이전 이미지 덮어쓰기)
        """
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.latest_image[camera_name] = (frame, msg)
        except Exception as exc:
            self.get_logger().warning(f'[{camera_name}] 이미지 디코딩 실패: {exc}')
    
    def _person_callback(self, msg: String, camera_name: str):
        """
        Person contours 콜백
        
        처리:
        - JSON 메시지 파싱
        - detections dict에서 contours 리스트 추출
        - 동적 마스크 생성을 위해 저장
        """
        try:
            data = json.loads(msg.data)
            detections = data.get('detections', {})
            # 새 형식: detections = {person_count: ..., boxes: [...], contours: [...]}
            if isinstance(detections, dict):
                contours = detections.get('contours', []) or []
            else:
                contours = detections if isinstance(detections, list) else []
            self.latest_person_contours[camera_name] = contours
        except Exception:
            self.latest_person_contours[camera_name] = []

    def _turtlebot_callback(self, msg: String, camera_name: str):
        """
        Turtlebot boxes 콜백
        
        처리:
        - JSON 메시지 파싱
        - detections dict에서 boxes 리스트 추출
        - 동적 마스크 생성을 위해 저장
        """
        try:
            data = json.loads(msg.data)
            detections = data.get('detections', {})
            # 새 형식: detections = {turtlebot_count: ..., boxes: [...]}
            if isinstance(detections, dict):
                boxes = detections.get('boxes', []) or []
            else:
                boxes = detections if isinstance(detections, list) else []
            self.latest_turtlebot_boxes[camera_name] = boxes
        except Exception:
            self.latest_turtlebot_boxes[camera_name] = []
    
    def _check_reset_signal(self):
        """
        파일 기반 reset 신호 체크
        
        목적:
        - UI 버튼 또는 외부 스크립트에서 경보 리셋 트리거
        - 터치된 파일을 감지하고 처리
        
        신호 파일:
        - /tmp/reset_alert.all: 모든 카메라 리셋
        - /tmp/reset_alert.cam01: cam01만 리셋
        
        UI에서 reset을 원하면:
        - 모든 카메라: touch /tmp/reset_alert.all
        - 특정 카메라: touch /tmp/reset_alert.cam01
        """
        # 모든 카메라 초기화 신호
        if os.path.exists('/tmp/reset_alert.all'):
            try:
                for camera in CAMERAS:
                    self._do_reset(camera)
                os.remove('/tmp/reset_alert.all')
                self.get_logger().info('✅ 파일 신호로 모든 카메라 Alert 초기화')
            except Exception as e:
                self.get_logger().warning(f'❌ 모든 카메라 초기화 실패: {e}')
        
        # 카메라별 초기화 신호
        for camera in CAMERAS:
            signal_file = f'/tmp/reset_alert.{camera}'
            if os.path.exists(signal_file):
                try:
                    self._do_reset(camera)
                    os.remove(signal_file)
                    self.get_logger().info(f'✅ 파일 신호로 [{camera}] Alert 초기화')
                except Exception as e:
                    self.get_logger().warning(f'❌ [{camera}] 초기화 실패: {e}')
    
    def _do_reset(self, camera_name: str):
        """
        카메라 경보 및 배경 초기화
        
        동작:
        1. 누적 차분 이미지 초기화
        2. 경보 상태 플래그 리셋
        3. 변화 버퍼 비우기
        4. 기준 프레임 재설정
        5. FALSE 경보 토픽 발행
        """
        self.accumulated_diff_images[camera_name] = None
        self.previous_collapse_state[camera_name] = False
        self.change_buffer[camera_name].clear()
        self._person_mask_history[camera_name].clear()
        
        # 기준 프레임 재설정
        if self.latest_image[camera_name] is not None:
            frame, msg_for_publish = self.latest_image[camera_name]
            self.diff_calculator.initialize_reference(camera_name, frame)
            # 즉시 FALSE 토픽 발행
            self._publish_alert(camera_name, msg_for_publish, False)
        else:
            # latest_image가 None이어도 FALSE alert 발행
            self._publish_alert_direct(camera_name, False)
        
        # 즉시 프레임 처리 (경쟁 조건 해결)
        if self.latest_image[camera_name] is not None:
            pass  # 이미 처리됨
    
    def _reset_residual_callback(self, msg: String):
        """Alert & 누적 차분 이미지 초기화 신호 처리 (Topic 방식 - 레거시)"""
        self.get_logger().info(f'🔄 [RESET_CALLBACK] 신호 수신: data={msg.data}')
        
        try:
            # "reset", "reset:all", "reset:cam01" 등 형식
            command = msg.data.strip().strip("'\"").lower()
            
            if command == 'reset' or command.startswith('reset:all'):
                # 모든 카메라 초기화
                for camera in CAMERAS:
                    self._do_reset(camera)
                self.get_logger().info('✅ Topic 신호로 모든 카메라 Alert 초기화')
            elif command.startswith('reset:'):
                # 특정 카메라만 초기화
                camera_name = command.split(':')[1].strip()
                if camera_name in CAMERAS:
                    self._do_reset(camera_name)
                    self.get_logger().info(f'✅ Topic 신호로 [{camera_name}] Alert 초기화')
                else:
                    self.get_logger().warning(f'❌ 잘못된 카메라: {camera_name}')
            else:
                self.get_logger().warning(f'❌ 인식할 수 없는 형식: {command}')
        except Exception as e:
            self.get_logger().warning(f'❌ Reset 신호 처리 실패: {e}')
    
    def _process_frame(self):
        """
        메인 프레임 처리 루프 (10ms 타이머)
        
        단계:
        1. 각 카메라의 최신 프레임 처리
        2. 동적 객체 마스크 생성
        3. 차분 계산
        4. 경보 판정
        5. 토픽 발행
        """
        for camera_name in CAMERAS:
            if self.latest_image[camera_name] is None:
                continue
            
            frame, msg = self.latest_image[camera_name]
            height, width = frame.shape[:2]
            
            # 해상도 초기 로깅
            if self.camera_resolutions[camera_name] is None:
                self.camera_resolutions[camera_name] = (width, height)
                self.get_logger().info(f'[{camera_name}] 실제 해상도: {width}x{height}')
            
            # 동적 객체 마스크 생성
            # 1) Person contour 마스크 (dilate=50px 여유)
            person_contour_mask = DynamicObjectMask.create_mask_from_contours(
                height,
                width,
                self.latest_person_contours[camera_name],
                dilate_pixels=50,
            )
            
            # 2) Person bounding box 마스크 (contour보다 넓은 보수적 커버)
            person_bbox_mask = DynamicObjectMask.create_mask_from_person_boxes(
                height,
                width,
                self.latest_person_contours[camera_name],
                padding=40,
            )
            
            # 3) Turtlebot bbox 마스크
            turtlebot_mask = DynamicObjectMask.create_mask_from_boxes(
                height,
                width,
                self.latest_turtlebot_boxes[camera_name],
            )
            
            # 4) 현재 프레임의 person 마스크를 히스토리에 추가
            current_person_mask = DynamicObjectMask.combine_masks(
                person_contour_mask, person_bbox_mask,
            )
            self._person_mask_history[camera_name].append(current_person_mask)
            
            # 5) 최근 N프레임의 person 위치 이력을 누적 (이전 위치도 마스킹)
            history_mask = np.zeros((height, width), dtype=np.uint8)
            for hist_mask in self._person_mask_history[camera_name]:
                if hist_mask is not None and hist_mask.shape == history_mask.shape:
                    history_mask = cv2.bitwise_or(history_mask, hist_mask)
            
            # 최종 동적 마스크 = person(현재+이력) + turtlebot
            dynamic_mask = DynamicObjectMask.combine_masks(
                history_mask, turtlebot_mask,
            )
            
            # 차분 계산 (ROI를 실제 프레임 크기에 맞게 동적 변환)
            roi_config = CAMERA_ROIS[camera_name]
            roi_ratio = roi_config.structure_roi
            
            # 비율 기반 ROI를 픽셀 좌표로 변환
            if isinstance(roi_ratio[0], float):
                # 비율 기반 (0.0 ~ 1.0)
                x1 = int(width * roi_ratio[0])
                y1 = int(height * roi_ratio[1])
                x2 = int(width * roi_ratio[2])
                y2 = int(height * roi_ratio[3])
            else:
                # 절대 좌표
                x1, y1, x2, y2 = roi_ratio
            
            actual_roi = (x1, y1, x2, y2)
            
            diff_result = self.diff_calculator.calculate_diff(
                camera_name,
                frame,
                dynamic_mask,
                actual_roi,
            )
            
            # 변화 버퍼에 추가
            self.change_buffer[camera_name].append(diff_result['has_change'])
            
            # 붕괴 경보 판정
            collapse_alert = self._judge_collapse(camera_name)
            
            # 중요: 이미 Alert 중이면 수동 초기화 전까지 계속 유지 (자동으로 FALSE로 돌아가지 않음)
            if self.previous_collapse_state[camera_name]:
                collapse_alert = True
            
            # Alert 새로 발생 (FALSE → TRUE)
            if not self.previous_collapse_state[camera_name] and collapse_alert:
                self.accumulated_diff_images[camera_name] = None  # 누적 초기화
                self.get_logger().warn(
                    f'[{camera_name}] ⚠️ COLLAPSE DETECTED! '
                    f"'r' 키로 초기화할 때까지 유지"
                )
            
            # Alert 유지 중
            if collapse_alert:
                # 차분 이미지를 누적 (OR 연산)
                diff_img = diff_result['diff_image']
                if diff_img is not None:
                    if self.accumulated_diff_images[camera_name] is None:
                        self.accumulated_diff_images[camera_name] = diff_img.copy()
                    else:
                        self.accumulated_diff_images[camera_name] = cv2.bitwise_or(
                            self.accumulated_diff_images[camera_name],
                            diff_img
                        )
            
            # Alert 없음
            elif not diff_result['has_change']:
                self.diff_calculator.update_reference(camera_name, frame)
            
            # 상태 변화할 때만 Alert 토픽 발행 (효율화)
            if self.previous_collapse_state[camera_name] != collapse_alert:
                self._publish_alert(camera_name, msg, collapse_alert)
            
            # Diff 이미지는 Alert 중에 계속 발행 (매 프레임)
            if collapse_alert and self.accumulated_diff_images[camera_name] is not None:
                self._publish_diff_image(camera_name, msg)
            
            # 이전 Alert 상태 저장
            self.previous_collapse_state[camera_name] = collapse_alert
    
    def _judge_collapse(self, camera_name: str) -> bool:
        """
        붕괴 경보 판정 로직
        
        알고리즘:
        - 최근 min_frames_to_alert 프레임을 확인
        - 그 중 60% 이상이 변화를 감지하면 True
        - 스로우 필터: 순간적인 노이즈에 강함
        
        반환:
        - True: 붕괴 가능성 감지
        - False: 정상
        """
        buffer = self.change_buffer[camera_name]
        
        # 카메라별 설정에서 min_frames_to_alert 가져오기
        cam_config = CAMERA_DETECTOR_CONFIG.get(camera_name, CAMERA_DETECTOR_CONFIG['cam01'])
        min_frames = cam_config['min_frames_to_alert']
        
        if len(buffer) < min_frames:
            return False
        
        # 최근 연속 프레임에서 변화 감지 비율
        recent = list(buffer)[-min_frames:]
        change_count = sum(recent)
        
        # 60% 이상 변화 감지 시 경보
        ratio = change_count / len(recent) if recent else 0.0
        return ratio > 0.6
    
    def _publish_alert(
        self,
        camera_name: str,
        msg: CompressedImage,
        collapse_detected: bool,
    ):
        """
        붕괴 경보 토픽 발행
        
        메시지 타입: JSON String
        내용: {camera_name, timestamp, collapse_detected, frames_in_buffer}
        """
        # Bool 메시지 발행
        msg_out = Bool()
        msg_out.data = collapse_detected
        self.alert_pub[camera_name].publish(msg_out)
    
    def _publish_alert_direct(
        self,
        camera_name: str,
        collapse_detected: bool,
    ):
        """
        붕괴 경보 토픽 발행 (CompressedImage 메시지 없을 때)
        
        사용처:
        - Reset 신호 처리 후 경보 상태 발행
        - 타임스탐프는 ROS clock에서 취득
        """
        # Bool 메시지 발행
        msg_out = Bool()
        msg_out.data = collapse_detected
        self.get_logger().info(f'[DEBUG] Alert 메시지 발행: {camera_name} = {collapse_detected}')
        self.alert_pub[camera_name].publish(msg_out)

    
    def _publish_diff_image(self, camera_name: str, msg: CompressedImage):
        """
        누적 차분 이미지 발행 (경보 중일 때만)
        
        목적:
        - Overlay node에서 붕괴 부위를 이미지로 표시
        - 경보 중에 계속 업데이트 (누적 차분)
        """
        if self.accumulated_diff_images[camera_name] is None:
            return
        
        display_img = self.accumulated_diff_images[camera_name]
        
        if len(display_img.shape) == 2:
            display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img_bgr = display_img
        
        _, jpeg_data = cv2.imencode('.jpg', display_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        msg_img = CompressedImage()
        msg_img.header.stamp = msg.header.stamp
        msg_img.format = 'jpeg'
        msg_img.data = jpeg_data.tobytes()
        
        self.diff_image_pub[camera_name].publish(msg_img)


def main(args=None):
    """
    메인 진입점
    
    ROS2 노드 시작:
    1. rclpy 초기화
    2. CollapseDetectorNode 생성
    3. 무한 스핀 (rclpy.spin)
    4. Ctrl+C로 종료
    """
    rclpy.init(args=args)
    node = None
    try:
        node = CollapseDetectorNode()
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

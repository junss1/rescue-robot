# camera_publisher.py v0.001 2026-03-16
# [이번 버전에서 수정된 사항]
# - 내장캠이 USB 카메라로 오인식되지 않도록 sysfs 장치 경로 기반 USB 판별을 추가
"""통합 카메라 발행기 - USB 웹캠 자동 감지 및 발행"""

import os
import sys
import io
import contextlib
import threading
import time

# OpenCV 경고 억제
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

import cv2
import rclpy
from typing import Optional, Any
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


@contextlib.contextmanager
def suppress_stderr():
    """
    stderr 임시 억제 (OpenCV 경고 메시지 제거용)
    
    사용:
    with suppress_stderr():
        cap = cv2.VideoCapture(device_index)
    """
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


def is_usb_camera(device_index: int) -> bool:
    """
    USB 카메라 여부 확인
    
    확인 단계:
    1. /dev/videoN 파일 존재 확인
    2. 프레임 읽기 성공 여부 확인
    3. 카메라 정보에서 내장/USB 판별
    
    반환:
    - True: USB 카메라 (외부 카메라)
    - False: 내장 웹캠 또는 사용 불가 카메라
    """
    try:
        # 1단계: /dev/videoN 파일 존재 확인
        video_path = f"/dev/video{device_index}"
        if not os.path.exists(video_path):
            return False
        
        # 2단계: 카메라 열기 및 프레임 읽기 성공 확인
        if not _can_read_frame(device_index):
            return False
        
        # 3단계: 내장 웹캠 여부 확인
        camera_info = _get_camera_info(device_index)
        
        if _is_usb_device_path(device_index):
            return True

        if camera_info is None:
            return False
        
        if _is_builtin_camera(camera_info):
            return False
        
        return _is_usb_camera_info(camera_info)
        
    except Exception:
        return False


def _can_read_frame(device_index: int) -> bool:
    """카메라에서 프레임 읽기 가능 여부"""
    try:
        with suppress_stderr():
            cap = cv2.VideoCapture(device_index)
            if not cap.isOpened():
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None
    except Exception:
        return False


def _get_camera_info(device_index: int) -> Optional[str]:
    """카메라 정보 조회 (/sys/class/video4linux에서)"""
    paths = [
        f"/sys/class/video4linux/video{device_index}/name",
        f"/sys/class/video4linux/video{device_index}/device/driver/module/name",
        f"/sys/class/video4linux/video{device_index}/device/driver/name",
    ]
    
    for path in paths:
        try:
            with open(path, 'r') as f:
                return f.read().strip().lower()
        except (FileNotFoundError, OSError):
            continue
    
    return None


def _get_device_syspath(device_index: int) -> Optional[str]:
    """카메라 장치의 sysfs 실제 경로 조회"""
    try:
        device_path = os.path.realpath(f"/sys/class/video4linux/video{device_index}/device")
    except OSError:
        return None
    return device_path if device_path else None


def _is_builtin_camera(camera_info: str) -> bool:
    """내장 웹캠 식별"""
    builtin_keywords = [
        'integrated', 'built-in', 'built in', 'internal',
        'laptop', 'notebook', 'webcam (integrated)', 'builtin'
    ]
    return any(keyword in camera_info for keyword in builtin_keywords)


def _is_usb_camera_info(camera_info: str) -> bool:
    """USB 카메라 정보 확인"""
    usb_keywords = ['usb', 'uvc', 'composite', 'web camera']
    return any(keyword in camera_info for keyword in usb_keywords)


def _is_usb_device_path(device_index: int) -> bool:
    """sysfs 장치 경로가 USB 버스 아래인지 확인"""
    device_syspath = _get_device_syspath(device_index)
    if device_syspath is None:
        return False
    return '/usb' in device_syspath.lower()


def find_usb_cameras(max_devices: int = 10, max_cameras: int = 2) -> dict:
    """
    연결된 USB 카메라 자동 감지
    
    기능:
    - /dev/videoN에서 사용 가능한 카메라 탐색
    - 내장 웹캠 제외
    - 메타데이터 스트림 제외
    - 지정된 개수만큼 감지 (기본 최대 2개)
    
    반환:
    - {'cam01': 0, 'cam02': 5, ...} 딕셔너리
    """
    cameras = {}
    camera_names = ['cam01', 'cam02', 'cam03', 'cam04']
    cam_idx = 0
    
    for device_index in range(max_devices):
        if cam_idx >= max_cameras:
            break
        
        try:
            if is_usb_camera(device_index):
                camera_name = camera_names[cam_idx]
                cameras[camera_name] = device_index
                cam_idx += 1
        except Exception:
            pass
        
        time.sleep(0.1)
    
    return cameras


class CameraConfig:
    """
    카메라 설정 상수
    
    프레임 레이트, 해상도, JPEG 품질, ROS2 QoS 등 카메라 관련 설정을 중앙집중식으로 관리합니다.
    """
    DEFAULT_FPS = 10                # 발행 프레임 레이트
    DEFAULT_WIDTH = 800             # 이미지 너비 (픽셀)
    DEFAULT_HEIGHT = 600            # 이미지 높이 (픽셀)
    DEFAULT_QUALITY = 90            # JPEG 압축 품질 (0-100)
    BUFFER_SIZE = 1                 # 카메라 버퍼 크기 (최신 프레임만 유지)
    WARMUP_FRAMES = 5               # 초기화 시 스킵할 프레임 수
    THREAD_TIMEOUT = 2              # 스레드 종료 대기 시간 (초)
    QOS_DEPTH = 1                   # ROS2 QoS 버퍼 크기


class CameraPublisherTask:
    """
    개별 USB 카메라를 처리하는 퍼블리셔 (독립적인 스레드에서 실행)
    
    기능:
    - USB 카메라 연결 및 설정 (해상도, FPS, 포맷)
    - ROS2 노드 생성 및 토픽 발행
    - 프레임 캡처 → JPEG 인코딩 → 발행
    - 스레드-안전한 리소스 정리
    """
    
    def __init__(self, camera_name: str, device_index: int, fps: int = 10, 
                 width: int = 800, height: int = 600, quality: int = 90):
        self.camera_name = camera_name
        self.device_index = device_index
        self.fps = fps
        self.width = width
        self.height = height
        self.jpeg_quality = quality
        self.topic_name = f'/camera/{camera_name}/raw'
        self.frame_id = f'{camera_name}_frame'
        
        # ROS2 및 카메라 리소스
        self.node = None
        self.publisher = None
        self.cap = None
        self.is_running = False
        self.thread = None
    
    def setup(self) -> None:
        """
        카메라와 ROS2 초기화
        
        순서:
        1. OpenCV를 통해 USB 카메라 연결
        2. ROS2 퍼블리셔 생성
        3. 카메라 워밍업 (첫 몇 프레임 스킵)
        """
        try:
            self._setup_camera()
            self._setup_ros2_publisher()
            self._warmup_camera()
        except Exception as e:
            self._log_error(f"초기화 실패: {e}")
            raise
    
    def _setup_camera(self) -> None:
        """
        카메라 열기 및 설정
        
        설정 항목:
        - MJPEG 포맷 (압축 효율)
        - 해상도 설정
        - FPS 설정
        - 버퍼 크기 최소화 (최신 프레임만 유지)
        """
        with suppress_stderr():
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                raise RuntimeError(f'카메라 device {self.device_index} 열기 실패')
        
        # 카메라 설정 (MJPEG + 해상도 + FPS + 버퍼)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CameraConfig.BUFFER_SIZE)
    
    def _setup_ros2_publisher(self) -> None:
        """
        ROS2 노드 및 퍼블리셔 생성
        
        각 카메라마다:
        - 독립적인 ROS2 노드 생성
        - RELIABLE QoS로 안정적 전송
        - CompressedImage 타입으로 발행
        
        토픽: /camera/{camera_name}/raw
        """
        self.node = Node(f'{self.camera_name}_pub')
        
        qos = QoSProfile(depth=CameraConfig.QOS_DEPTH)
        qos.reliability = ReliabilityPolicy.RELIABLE
        
        self.publisher = self.node.create_publisher(
            CompressedImage,
            self.topic_name,
            qos
        )
    
    def _warmup_camera(self) -> None:
        """카메라 워밍업 (첫 프레임들 스킵)"""
        for _ in range(CameraConfig.WARMUP_FRAMES):
            self.cap.read()
    
    def publish_frame(self) -> bool:
        """
        프레임 읽기, 인코딩, 발행
        
        반환:
        - True: 성공
        - False: 카메라 읽기 실패 또는 인코딩 실패
        """
        frame = self._read_frame()
        if frame is None:
            return False
        
        jpeg_buffer = self._encode_jpeg(frame)
        if jpeg_buffer is None:
            return False
        
        self._publish_message(jpeg_buffer)
        return True
    
    def _read_frame(self) -> Optional[Any]:
        """카메라에서 프레임 읽기"""
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def _encode_jpeg(self, frame: Any) -> Optional[bytes]:
        """프레임을 JPEG로 인코딩"""
        ok, buffer = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        return buffer.tobytes() if ok else None
    
    def _publish_message(self, jpeg_data: bytes) -> None:
        """
        ROS2 CompressedImage 메시지 발행
        
        메시지 구성:
        - header: 타임스탐프 및 프레임 ID
        - format: 'jpeg'
        - data: JPEG 바이너리 데이터
        """
        msg = CompressedImage()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = jpeg_data
        self.publisher.publish(msg)
    
    def run(self) -> None:
        """
        메인 루프 (스레드에서 실행)
        
        동작:
        - 무한 루프에서 프레임 캡처
        - JPEG 인코딩
        - ROS2 토픽 발행
        - FPS 유지를 위해 sleep 적용
        """
        self.setup()
        self.is_running = True
        frame_interval = 1.0 / self.fps
        
        while self.is_running:
            try:
                start_time = time.time()
                self.publish_frame()
                elapsed = time.time() - start_time
                
                # FPS 유지
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self._log_error(f"루프 오류: {e}")
                break
    
    def stop(self) -> None:
        """
        리소스 정리 (카메라 + ROS2)
        
        순서:
        1. is_running 플래그 해제
        2. 스레드 종료 대기
        3. 카메라 연결 해제
        4. ROS2 노드 파괴
        """
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=CameraConfig.THREAD_TIMEOUT)
        
        self._cleanup_camera()
        self._cleanup_ros2()
    
    def _cleanup_camera(self) -> None:
        """카메라 리소스 해제"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
    
    def _cleanup_ros2(self) -> None:
        """ROS2 노드 정리"""
        if self.node:
            self.node.destroy_node()
    
    def _log_error(self, message: str) -> None:
        """에러 로깅"""
        print(f"[{self.camera_name}] {message}", flush=True)
    
    def start(self) -> None:
        """데몬 스레드 시작"""
        self.thread = threading.Thread(target=self.run, daemon=False)
        self.thread.start()


def main(args=None) -> None:
    """
    메인 진입점
    
    동작:
    1. ROS2 초기화
    2. USB 카메라 자동 감지 (내장 웹캠 제외)
    3. 카메라마다 독립적인 퍼블리셔 스레드 시작
    4. Ctrl+C로 종료할 때까지 대기
    5. 모든 리소스 정리하고 ROS2 종료
    """
    rclpy.init(args=args)
    
    publishers = {}
    try:
        # USB 카메라 자동 감지
        cameras = find_usb_cameras(max_devices=10, max_cameras=2)
        if not cameras:
            print("[ERROR] USB 카메라를 찾을 수 없습니다", flush=True)
            return
        
        # 각 카메라마다 퍼블리셔 스레드 실행
        publishers = _start_camera_publishers(cameras)
        
        # 메인 스레드에서 대기 (Ctrl+C로 종료)
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n[INFO] 사용자가 종료 요청함", flush=True)
    except Exception as e:
        print(f"[ERROR] 예상치 못한 오류: {e}", flush=True)
    
    finally:
        if publishers:
            _stop_all_publishers(publishers)
        if rclpy.ok():
            rclpy.shutdown()


def _start_camera_publishers(cameras: dict) -> dict:
    """
    카메라별 퍼블리셔 생성 및 시작
    
    인자:
    - cameras: {카메라_이름: device_index} 딕셔너리
    
    반환:
    - {카메라_이름: CameraPublisherTask} 딕셔너리
    """
    publishers = {}
    
    for camera_name, device_index in cameras.items():
        try:
            pub = CameraPublisherTask(
                camera_name=camera_name,
                device_index=device_index,
                fps=CameraConfig.DEFAULT_FPS,
                width=CameraConfig.DEFAULT_WIDTH,
                height=CameraConfig.DEFAULT_HEIGHT,
                quality=CameraConfig.DEFAULT_QUALITY
            )
            pub.start()
            publishers[camera_name] = pub
            time.sleep(0.5)
            print(f"[INFO] {camera_name} 발행기 시작 (device {device_index})", flush=True)
        except Exception as e:
            print(f"[ERROR] {camera_name} 초기화 실패: {e}", flush=True)
    
    return publishers


def _stop_all_publishers(publishers: dict) -> None:
    """
    모든 카메라 퍼블리셔 정리
    
    각 퍼블리셔의 스레드를 종료하고 리소스를 해제합니다.
    """
    for camera_name, pub in publishers.items():
        try:
            pub.stop()
            print(f"[INFO] {camera_name} 정리 완료", flush=True)
        except Exception as e:
            print(f"[ERROR] {camera_name} 정리 실패: {e}", flush=True)


if __name__ == '__main__':
    main()

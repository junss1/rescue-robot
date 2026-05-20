# v0.620
# file: srd_pose_emergency_core.py
# date: 2026-03-11
# changes:
# - 세션 종료용 reset() 추가
# - 대표점 픽셀(rep_point_px) 및 보조 중심점 export 추가
# - 관제노드 victim_position 정확도 향상용 method 필드 추가
"""
SRD Pose EmergencyLevel Core
======================

이 파일은 ROS2 의존성이 없는 "분석 엔진" 파일이다.
입력으로 OpenCV BGR 프레임(np.ndarray)을 받고,
사람 포즈를 분석해서 다음 두 가지를 반환한다.

1) 시각화가 그려진 annotated frame
2) 구조화된 결과(result list)

즉, 이 파일은 카메라/토픽/네트워크를 몰라도 된다.
오직 "프레임 -> 포즈 분석 -> 상태 판정"만 담당한다.

실제 TurtleBot4 환경에서는 별도의 ROS2 노드가 이 코어를 import 해서 사용한다.
"""

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class AnalyzerConfig:
    """분석기에 필요한 모든 파라미터를 한곳에 모아둔 설정 클래스.

    코드 리뷰/튜닝 시 가장 먼저 보는 영역이다.
    현장에서 임계값을 바꾸고 싶을 때 이 블록만 수정하면 된다.
    """

    # ---------------------------------------------------------------------
    # Model / Input 관련 파라미터
    # ---------------------------------------------------------------------
    model_path: str = "yolo11n-pose.pt"  # YOLO Pose 모델 파일 경로
    det_conf: float = 0.35               # 사람 탐지 최소 confidence
    kp_conf_th: float = 0.45             # keypoint를 유효하다고 볼 최소 confidence
    kp_margin_px: int = 2                # 프레임 가장자리 keypoint 무시용 margin

    # ---------------------------------------------------------------------
    # Observation(관측 상태) 판단 파라미터
    # ---------------------------------------------------------------------
    low_conf_min_kps: int = 3            # 전체 keypoint가 이 개수보다 적으면 LOW_CONF
    upper_body_min_kps: int = 3          # 상체 keypoint가 이 개수 이상이면 UPPER_BODY 후보
    full_body_extra_lower_kps: int = 1   # hip 외 하체 keypoint가 이 개수 이상이면 FULL_BODY 후보

    # ---------------------------------------------------------------------
    # Posture(자세) 판단 파라미터
    # ---------------------------------------------------------------------
    # shoulder_tilt는 0~90도 범위의 예각으로 정규화된 어깨선 기울기다.
    leaning_shoulder_tilt_deg: float = 25.0
    collapsed_shoulder_tilt_deg: float = 45.0

    # head_drop_ratio는 0~1 범위를 갖는 값이다.
    # 0.0에 가까울수록 얼굴이 어깨보다 충분히 위에 있다.
    # 1.0에 가까울수록 얼굴이 어깨에 가까워지거나 아래로 내려온 상태다.
    leaning_head_drop_ratio: float = 0.55
    collapsed_head_drop_ratio: float = 0.78

    # 전신이 보일 때만 사용하는 torso / lying 관련 기준
    collapsed_torso_angle_deg: float = 60.0
    lying_torso_angle_deg: float = 72.0
    lying_aspect_ratio: float = 2.00

    # 상반신만 보일 때 어깨 간 가로폭이 너무 좁으면 tilt를 신뢰하지 않는다.
    upper_body_min_shoulder_span_ratio: float = 0.25

    # ---------------------------------------------------------------------
    # Motion(움직임) 판단 파라미터
    # ---------------------------------------------------------------------
    motion_window: int = 12              # 최근 N프레임 평균으로 움직임 smoothing
    motion_active_smooth: float = 0.020  # 전체적인 움직임이 이 값 이상이면 ACTIVE
    motion_active_upper: float = 0.025   # 상체 움직임이 이 값 이상이면 ACTIVE
    motion_local_only_upper: float = 0.015
    motion_local_only_core: float = 0.010
    motion_low: float = 0.008            # ACTIVE 미만이지만 이 값 이상이면 LOW

    # ---------------------------------------------------------------------
    # 상태 지속 시간 기준
    # ---------------------------------------------------------------------
    analyzing_sec: float = 1.5
    caution_sec: float = 4.5
    warning_sec: float = 5.5
    critical_sec: float = 7.0

    # ---------------------------------------------------------------------
    # 시각화 옵션
    # ---------------------------------------------------------------------
    show_debug: bool = True              # 디버그 텍스트(tilt/hds/m) 표시 여부
    draw_skeleton: bool = True           # skeleton 시각화 여부
    draw_box: bool = True                # bbox 시각화 여부


class PoseEmergencyEngine:
    """사람 포즈(행동) 기반 위급 상태 판정 핵심 엔진 시스템.

    이 클래스는 ROS2 노드와 완전히 분리되어 독립적으로 동작하는 코어 로직입니다.
    다음과 같은 7단계 파이프라인으로 구성되어 있습니다:
    1) YOLO Pose 추론을 통해 사람의 바운딩 박스와 17개 관절(Keypoint) 좌푯값을 추출
    2) 신뢰도(Confidence)와 이미지 경계선을 기준으로 유효한 Keypoint만 필터링
    3) 신체가 얼마나 가려졌는지 가시성을 판단 (전신 / 상반신 / 일부 노출 / 인식 불가)
    4) 어깨 기울기, 고개 꺾임 비율, 허리 숙임 각도를 계산하여 5가지 자세 상태(Posture)로 분류
    5) 프레임 간 관절 이동량을 계산하여 4단계 움직임(Motion) 크기로 분류
    6) 위의 결과들과 각 상태가 얼마나 오래 지속되었는지를 종합해 최종 위급 단계(Emergency Level) 5단계로 판정
    7) 사람이 이해하기 쉽도록 원본 이미지 위에 뼈대, 박스, 상태 텍스트를 그린Annotated 프레임 생성
    """

    # COCO 데이터셋 기준 17개 관절(Keypoint)의 인덱스 번호 정의
    UPPER_IDS = [0, 5, 6, 7, 8, 9, 10]    # [코, 왼쪽 어깨, 오른쪽 어깨, 왼쪽 팔꿈치, 오른쪽 팔꿈치, 왼쪽 손목, 오른쪽 손목] => 상체 주요 관절
    LOWER_IDS = [11, 12, 13, 14, 15, 16]  # [왼쪽 골반, 오른쪽 골반, 왼쪽 무릎, 오른쪽 무릎, 왼쪽 발목, 오른쪽 발목] => 하체 주요 관절
    CORE_IDS = [5, 6, 11, 12]             # [양 어깨, 양쪽 골반] => 몸통(코어) 부위, 이 4점이 고정되어야 전체적인 큰 움직임으로 인식

    # 시각화할 때 상반신 관절들을 선으로 묶기 위한 연결 고리 (idx1, idx2)
    UPPER_LINKS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)] # 어깨끼리, 어깨-팔꿈치-손목 연결

    # 시각화할 때 전신 관절들을 선으로 묶기 위한 연결 고리
    # 상체 연결 고리에 하체(골반, 무릎, 발목) 연결을 추가 합산함
    FULL_LINKS = UPPER_LINKS + [
        (5, 11), (6, 12), (11, 12),       # 상하체 연결(어깨-골반) 및 골반끼리 연결
        (11, 13), (13, 15),               # 왼쪽 골반-무릎-발목
        (12, 14), (14, 16),               # 오른쪽 골반-무릎-발목
    ]

    # emergency_level 시각화 색상(BGR)
    COLORS = {
        "ANALYZING": (255, 255, 255),
        "NORMAL": (0, 200, 0),
        "CAUTION": (0, 220, 255),
        "WARNING": (0, 140, 255),
        "CRITICAL": (0, 0, 255),
    }

    # emergency_level 우선순위. 여러 사람이 잡히더라도 가장 높은 위험도를 대표값으로 사용.
    EMERGENCY_PRIORITY = {
        "CRITICAL": 4,
        "WARNING": 3,
        "CAUTION": 2,
        "NORMAL": 1,
        "ANALYZING": 0,
    }

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.cfg = config or AnalyzerConfig()
        self.model = YOLO(self.cfg.model_path)

        # track_id 별 히스토리 저장소
        # - first_seen: 처음 본 시간
        # - prev_kps / prev_conf: 이전 프레임 keypoint, confidence
        # - motion_buf: 최근 움직임 평균 버퍼
        # - last_signature / state_since: 같은 상태가 얼마나 지속되었는지 계산용
        self.history: Dict[int, dict] = {}

    def reset(self) -> None:
        """세션 종료 시 track history를 초기화한다."""
        self.history.clear()

    # ------------------------------------------------------------------
    # 기본 유틸리티 함수: 뼈대 좌표를 수학적으로 정제하는 핵심 기초 함수들
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_mean(points: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """주어진 좌표들 중에서, 값이 존재하는(None이 아닌) 유효한 좌표들만 골라 평균점을 반환합니다.

        활용 예시:
        - 왼쪽 어깨와 오른쪽 어깨의 _safe_mean() -> 목(어깨 중심점) 좌표 도출
        - 눈, 코, 귀 좌표 5개의 _safe_mean() -> 얼굴의 대략적인 무게 중심점(Face Anchor) 도출
        한쪽 팔이나 눈이 카메라에 안 보여서 None이 들어와도 에러 없이 계산해주는 안전 장치입니다.
        """
        valid = [p for p in points if p is not None]
        return np.mean(valid, axis=0) if valid else None

    @staticmethod
    def _pt_to_list(point: Optional[np.ndarray]) -> Optional[List[int]]:
        """numpy point -> JSON-safe [x, y] 변환."""
        if point is None:
            return None
        return [int(round(float(point[0]))), int(round(float(point[1])))]

    def _extract_rep_points(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        box: np.ndarray,
        visibility: str,
        shape: Tuple[int, int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        관제노드 좌표 추정용 대표 픽셀점을 계산한다.

        반환:
        - rep_point
        - shoulder_center
        - hip_center
        - face_anchor
        - rep_point_method
        """
        nose = self._get_point(keypoints, kp_conf, 0, shape)
        leye = self._get_point(keypoints, kp_conf, 1, shape)
        reye = self._get_point(keypoints, kp_conf, 2, shape)
        lear = self._get_point(keypoints, kp_conf, 3, shape)
        rear = self._get_point(keypoints, kp_conf, 4, shape)
        ls = self._get_point(keypoints, kp_conf, 5, shape)
        rs = self._get_point(keypoints, kp_conf, 6, shape)
        lh = self._get_point(keypoints, kp_conf, 11, shape) if visibility == "FULL_BODY" else None
        rh = self._get_point(keypoints, kp_conf, 12, shape) if visibility == "FULL_BODY" else None

        shoulder_center = self._safe_mean([ls, rs])
        hip_center = self._safe_mean([lh, rh])
        face_anchor = self._safe_mean([nose, leye, reye, lear, rear])

        rep_point = None
        rep_method = "NONE"

        if visibility == "FULL_BODY" and shoulder_center is not None and hip_center is not None:
            rep_point = self._safe_mean([shoulder_center, hip_center])
            rep_method = "SHOULDER_HIP_MID"
        elif visibility == "UPPER_BODY" and shoulder_center is not None:
            rep_point = shoulder_center
            rep_method = "SHOULDER_CENTER"
        elif face_anchor is not None:
            rep_point = face_anchor
            rep_method = "FACE_ANCHOR"
        else:
            x1, y1, x2, y2 = box.astype(float)
            rep_point = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            rep_method = "BBOX_CENTER"

        return rep_point, shoulder_center, hip_center, face_anchor, rep_method

    def _is_valid_kp(self, point: np.ndarray, conf: float, w: int, h: int) -> bool:
        """한 관절(Keypoint)이 분석에 쓸 수 있는 "신뢰할 만한" 좌표인지 검사합니다.

        신뢰 판단 조건 두 가지 (하나라도 실패하면 False):
        1) 화면 가장자리에 너무 붙어있지 말 것 (오른쪽/아래 끝에서 화면이 잘렸을 때의 오작동 방지용 kp_margin_px 검사)
        2) YOLO 모델이 계산한 확신도(confidence)가 최소 기준치(kp_conf_th, 기본 0.45) 이상일 것
        """
        x, y = float(point[0]), float(point[1])
        m = self.cfg.kp_margin_px
        return (
            m <= x < (w - m)                    # X좌표가 좌우 여백 안에 들어오는지
            and m <= y < (h - m)                # Y좌표가 상하 여백 안에 들어오는지
            and float(conf) >= self.cfg.kp_conf_th  # AI의 확신도가 기준을 넘었는지
        )

    def _valid_indices(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        ids: List[int],
        shape: Tuple[int, int, int],
    ) -> List[int]:
        """주어진 관절 인덱스 번호들 중에서 '진짜 쓸 수 있는 유효한 관절'의 번호들만 걸러서 반환합니다."""
        h, w = shape[:2]
        # _is_valid_kp 필터링을 통과한 번호만 리스트 패킹
        return [i for i in ids if self._is_valid_kp(keypoints[i], kp_conf[i], w, h)]

    def _get_point(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        idx: int,
        shape: Tuple[int, int, int],
    ) -> Optional[np.ndarray]:
        """해당 특정 관절(idx)이 유효하다면 [x, y] 좌푯값을 주고, 유효하지 않으면 None을 던져줍니다."""
        h, w = shape[:2]
        return keypoints[idx] if self._is_valid_kp(keypoints[idx], kp_conf[idx], w, h) else None

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        """두 좌표(점)를 이은 1차원 선분이 수평선으로부터 얼마나 기울어졌는지 각도(절댓값)를 계산합니다.

        [주의할 점]
        YOLO 모델은 사람 기준 오른쪽 어깨를 항상 '오른쪽'으로 부릅니다.
        하지만 사람이 우리를 등지고 돌아서 있을 경우 2D 카메라 좌표상에서는 오른쪽 어깨의 X축이 더 왼쪽에 있을 수 있습니다.
        이런 3D->2D 투시 역전 때문에 단순 아크탄젠트(atan2)를 구하면 170도가 나오거나 -10도가 나오는 등 각도가 요동칩니다.
        구조 관제에서는 사람이 쓰러진 '방향' 보다는 뼈대가 기울어진 '정도' 자체가 더 위급함을 내포하므로,
        수학적 처리를 통해 항상 90도를 넘지 않는 **예각(0~90도)** 사이클로 정규화(Normalize)해서 씁니다.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        angle = abs(math.degrees(math.atan2(dy, dx)))  # 일단 0~180도로 절댓값을 뽑아냅니다.
        
        # 선은 양방향이 위상 동형이므로, 90도가 넘어간 각도는 보색처럼 반전시켜 예각(0~90)으로 만듭니다.
        if angle > 90.0:
            angle = 180.0 - angle
        return angle

    def _new_track_state(self) -> dict:
        """처음 카메라에 찍힌 '새로운 사람'에게 빈 기억 공간(Memory Dictionary)을 만들어 줍니다.
        
        로봇이 움직이며 여러 사람을 추적할 수 있으므로, 각 사람(track_id)마다 독립된 이력(History)을 가져야 합니다.
        """
        now = time.time()
        return {
            "first_seen": now,                        # 이 사람을 처음 만난 유닉스 타임 (시간 기반 분석용)
            "prev_kps": None,                         # 이전 프레임의 관절 위치 전체 배열 (동작 크기 비교용)
            "prev_conf": None,                        # 이전 프레임의 AI 확신도 배열
            "motion_buf": deque(maxlen=self.cfg.motion_window), # 최근 12프레임 정도의 움직임 양을 담아둘 큐 (흔들림 방지, 1초 단위 스무딩용)
            "last_signature": None,                   # 직전 프레임에서 이 사람이 받은 상태 진단명 (예: "FULL_BODY|NORMAL|NONE")
            "state_since": now,                       # 그 진단명이 안 바뀌고 지속된 시간 시작점
        }

    # ------------------------------------------------------------------
    # 1) 가시성(Visibility) 분류: 카메라에 사람이 얼마나 잘 보이는가?
    # ------------------------------------------------------------------
    def _classify_visibility(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        shape: Tuple[int, int, int],
    ) -> str:
        """사람의 몸이 카메라에 얼마나 노출되었는지 4단계 가시성 상태로 분류합니다.

        [반환값 의미]
        - LOW_CONF   : 신뢰할 수 있는 관절점이 너무 적어 분석 불가 (오탐지이거나 너무 멀리 있음)
        - FULL_BODY  : 상체와 하체가 모두 충분히 확보된 상태 (가장 정확한 전신 분석 가능)
        - UPPER_BODY : 상체는 잘 보이지만 하체가 가려지거나 잘린 상태 (상반신 분석 모드로 동작)
        - PARTIAL    : 상하체 전체적으로 관절이 부족하지만 인식 무시는 아닌 상태 (일부 노출)
        """
        # 먼저 상체와 하체에서 '유효한(믿을만한)' 관절들의 번호표만 모아옵니다.
        upper_valid = self._valid_indices(keypoints, kp_conf, self.UPPER_IDS, shape)
        lower_valid = self._valid_indices(keypoints, kp_conf, self.LOWER_IDS, shape)
        total_count = len(upper_valid) + len(lower_valid)

        # 전체 유효 관절 수가 설정된 최소치(기본 3개)보다 적으면 '보이지 않음'으로 간주
        if total_count < self.cfg.low_conf_min_kps:
            return "LOW_CONF"

        # [전신 FULL_BODY 판단 조건]
        # 1) 상체 관절이 충분히 보임 (upper_body_min_kps 이상)
        # 2) 왼쪽(11) 혹은 오른쪽(12) 골반 중 하나는 반드시 보여야 함 (상하체 연결점)
        # 3) 골반을 제외한 나머지 하체 관절(무릎, 발목)이 일정 개수 이상 추가로 보여야 함
        hips_ok = 11 in lower_valid or 12 in lower_valid
        extra_lower = len([i for i in lower_valid if i not in (11, 12)]) # 골반 제외 하체 관절 개수

        if (
            len(upper_valid) >= self.cfg.upper_body_min_kps       # 상체 조건 만족
            and hips_ok                                           # 골반 뼈 존재 확인
            and extra_lower >= self.cfg.full_body_extra_lower_kps # 추가 하체 관절 확인
        ):
            return "FULL_BODY"

        # 전신 조건은 만족하지 못했지만, 상체 관절 개수만 충분하다면 상반신으로 판정
        if len(upper_valid) >= self.cfg.upper_body_min_kps:
            return "UPPER_BODY"

        # 상체도 하체도 기준에 못 미치지만, 완전 LOW_CONF는 아닌 애매한 일부 노출 상태
        return "PARTIAL"

    # ------------------------------------------------------------------
    # 2) 자세(Posture) 분류: 사람의 몸이 물리적으로 어떤 각도인가?
    # ------------------------------------------------------------------
    def _classify_posture(
        self,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        box: np.ndarray,
        visibility: str,
        shape: Tuple[int, int, int],
    ) -> Tuple[str, float, float, float]:
        """사람의 관절 기울기를 삼각함수로 계산하여 현재 자세를 분류합니다.

        [분류 단계]
        - NORMAL(정상) : 특별히 기울어지거나 꺾이지 않은 일반적인 서거나 앉은 자세
        - LEANING(기울어짐) : 어깨나 고개가 살짝 꺾이거나 숙여진 불안정한 자세
        - COLLAPSED(쓰러짐/붕괴) : 뼈대가 극심하게 꺾여 심각한 부상이나 기절이 의심되는 자세
        - LYING(누움) : 몸통 전체가 바닥과 수평에 가깝게 누워있는 자세 (전신 노출일 때만 판정)
        - UNKNOWN(알 수 없음) : 가시성이 확보되지 않아 자세를 파악할 수 없는 상태

        [반환값 튜플]
        - (자세 텍스트, 어깨 기울기 각도, 머리 꺾임 비율, 허리 숙임 각도)
        """
        # 바운딩 박스를 통해 사람 박스의 넓이(bw), 높이(bh), 그리고 가로/세로 비율(aspect)을 잽니다.
        x1, y1, x2, y2 = box.astype(float)
        bw = max(x2 - x1, 1.0) # 박스 넓이 (0으로 나누는 에러를 막기 위해 최소 1.0 설정)
        bh = max(y2 - y1, 1.0) # 박스 높이
        aspect = bw / bh       # 비율이 클수록(가로로 넓을수록) 사람이 누워있을 확률이 높습니다.

        # 얼굴 / 상체 / 하체 주요 관절점(Keypoint)들을 전부 불러와 준비합니다.
        nose = self._get_point(keypoints, kp_conf, 0, shape)
        leye = self._get_point(keypoints, kp_conf, 1, shape)
        reye = self._get_point(keypoints, kp_conf, 2, shape)
        lear = self._get_point(keypoints, kp_conf, 3, shape)
        rear = self._get_point(keypoints, kp_conf, 4, shape)
        ls = self._get_point(keypoints, kp_conf, 5, shape)
        rs = self._get_point(keypoints, kp_conf, 6, shape)

        # 골반(Hip) 점은 '전신'이 다 보일 때만 사용합니다. 상반신 모드일 때는 억지로 추정하지 않습니다.
        lh = self._get_point(keypoints, kp_conf, 11, shape) if visibility == "FULL_BODY" else None
        rh = self._get_point(keypoints, kp_conf, 12, shape) if visibility == "FULL_BODY" else None

        # 양쪽 어깨, 양쪽 골반, 얼굴 이목구비 5점들의 '중간점(무게중심)'을 각각 구합니다.
        shoulder_center = self._safe_mean([ls, rs])
        hip_center = self._safe_mean([lh, rh])
        face_anchor = self._safe_mean([nose, leye, reye, lear, rear])

        # 좌우 어깨 점을 이어 '어깨선이 기울어진 각도'를 계산합니다. (0~90도)
        shoulder_tilt = self._angle_deg(ls, rs) if ls is not None and rs is not None else 0.0

        # 어깨선 가로 길이(픽셀 단위 너비) 측정
        shoulder_span = 0.0
        if ls is not None and rs is not None:
            shoulder_span = abs(float(rs[0] - ls[0]))

        # [측면(옆보기) 방어 로직] 
        # 사람이 몸을 카메라 옆으로 심하게 돌려 서면 2D 상에서는 양 어깨가 겹쳐서 어깨 너비가 확 줄어듭니다.
        # 이 상태에서 고개만 살짝 까딱해도 3D->2D 투영 왜곡 때문에 각도가 엄청 심하게 꺾인 것처럼 오계산됩니다.
        # 따라서 어깨 너비가 사람 박스 전체 가로폭 대비 너무 좁다면(옆모습이라면) 어깨 기울기 값을 아예 무시(0.0)해버립니다.
        if shoulder_span < bw * self.cfg.upper_body_min_shoulder_span_ratio:
            shoulder_tilt = 0.0

        # [고개 꺾임 비율 (head_drop_ratio) 계산 로직]
        # 얼굴의 무게중심(face_anchor)이 어깨 중심선(shoulder_center)에 얼마나 가깝게 밀착되었는지를 봅니다.
        # 목이 꼿꼿하면 거리가 멀리 떨어져 있고(0.0에 가까움), 기절해서 목이 꺾이거나 책상에 엎드리면 거리가 좁아집니다(1.0에 가까움).
        head_drop_ratio = 0.0
        if face_anchor is not None and shoulder_center is not None and shoulder_span > 1.0:
            # 어깨 Y좌표에서 얼굴 Y좌표를 빼서 수직 거리 차이를 잰 후 (음수 방어용 max 0.0)
            head_gap = max(float(shoulder_center[1] - face_anchor[1]), 0.0)
            # 수직 거리를 어깨뼈 너비(개인 신체 비율 기준)로 나눠서 체형에 상관없이 0~1사이 정규화된 꺾임 비율 도출
            head_drop_ratio = 1.0 - min(head_gap / shoulder_span, 1.0)

        torso_angle = 0.0
        if shoulder_center is not None and hip_center is not None:
            # 척추(어깨-골반을 잇는 선)가 곧게 선 수직 기준선(90도)에서 얼마나 벗어났는지 각도 편차를 구합니다.
            torso_angle = abs(90.0 - self._angle_deg(shoulder_center, hip_center))

        # ------------------------------
        # [전신 FULL_BODY 일 때의 자세 판정 규칙]
        # ------------------------------
        if visibility == "FULL_BODY":
            # [LYING - 누움]
            # 1) 전체 바운딩 박스가 사람 키보다 가로로 2배 이상 길쭉해진 경우 (aspect >= 2.00)
            # 2) 또는 척추가 거의 바닥과 수평(torso_angle >= 72도)인 경우 완벽히 누웠다고 봅니다.
            if aspect >= self.cfg.lying_aspect_ratio or torso_angle >= self.cfg.lying_torso_angle_deg:
                return "LYING", shoulder_tilt, head_drop_ratio, torso_angle

            # [COLLAPSED - 쓰러짐/붕괴]
            # 위 누움 단계까진 아니지만, 다음 3개 중 하나라도 심각하게 꺾인 경우 의식을 잃고 쓰러졌다고 간주합니다.
            # 1) 허리가 60도 이상 꺾임 / 2) 어깨가 45도 이상 꺾임 / 3) 고개가 어깨팍에 거의 파묻힘(78% 이상)
            if (
                torso_angle >= self.cfg.collapsed_torso_angle_deg
                or shoulder_tilt >= self.cfg.collapsed_shoulder_tilt_deg
                or head_drop_ratio >= self.cfg.collapsed_head_drop_ratio
            ):
                return "COLLAPSED", shoulder_tilt, head_drop_ratio, torso_angle

            # [LEANING - 기울어짐]
            # 위 붕괴 단계까진 아니지만, 정상적으로 서있지 못하고 자세가 무너지는 중(Leaning)으로 봅니다.
            # 1) 어깨선이 25도 이상 기울어짐 / 2) 고개가 어깨 쪽으로 눈에 띄게(55% 이상) 내려옴
            if (
                shoulder_tilt >= self.cfg.leaning_shoulder_tilt_deg
                or head_drop_ratio >= self.cfg.leaning_head_drop_ratio
            ):
                return "LEANING", shoulder_tilt, head_drop_ratio, torso_angle

            # 위 모든 위급 조건을 전부 피해갔다면 (꼿꼿이 서있거나, 바르게 앉은 다소곳한 모양새)
            return "NORMAL", shoulder_tilt, head_drop_ratio, torso_angle

        # ------------------------------
        # [상반신 UPPER_BODY 일 때의 자세 판정 규칙]
        # ------------------------------
        if visibility == "UPPER_BODY":
            # 상반신만 보일 때는 골반(Hip) 위치를 모르므로 허리 각도(torso_angle)는 쓰지 못합니다.
            # 어깨(shoulder_tilt)와 고개(head_drop_ratio) 단 두 가지의 상태만으로 자세를 추론합니다.

            # [COLLAPSED - 붕괴] 어깨나 고개 중 하나라도 기준치(45도, 78%) 이상 심하게 꺾이면 의심
            if (
                shoulder_tilt >= self.cfg.collapsed_shoulder_tilt_deg
                or head_drop_ratio >= self.cfg.collapsed_head_drop_ratio
            ):
                return "COLLAPSED", shoulder_tilt, head_drop_ratio, torso_angle

            # [LEANING - 기울어짐] 어깨나 고개가 기준치(25도, 55%) 이상 불안정하게 기운 경우
            if (
                shoulder_tilt >= self.cfg.leaning_shoulder_tilt_deg
                or head_drop_ratio >= self.cfg.leaning_head_drop_ratio
            ):
                return "LEANING", shoulder_tilt, head_drop_ratio, torso_angle

            # 안전
            return "NORMAL", shoulder_tilt, head_drop_ratio, torso_angle

        # 전신도 상반신도 아니면(PARTIAL 등) 관절 개수가 모자라므로, 자세를 '모름(UNKNOWN)'으로 예외처리
        return "UNKNOWN", shoulder_tilt, head_drop_ratio, torso_angle

    # ------------------------------------------------------------------
    # 3) 움직임(Motion) 계산 / 분류: 이 사람이 움직이고 있는가?
    # ------------------------------------------------------------------
    def _motion_value(
        self,
        track_id: int,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        box: np.ndarray,
        shape: Tuple[int, int, int],
    ) -> Tuple[float, float, float]:
        """바로 이전 프레임의 관절 좌표를 꺼내와 현재 프레임과 비교해, 픽셀 단위의 이동량(움직임 크기)을 계산합니다.

        [반환값 튜플]
        - smooth : 최근 N개(motion_window) 프레임 동안의 평균적인 전체 최대 움직임 
        - upper  : 이번 프레임에서 측정한 '팔/머리 등 상체' 관절들의 평균 이동 거리
        - core   : 이번 프레임에서 측정한 '어깨/골반 등 몸통코어' 관절들의 평균 이동 거리 (몸통이 고정되어있나 여부)
        """
        # 이 사람(track_id)의 히스토리 딕셔너리를 꺼내거나, 새로 만듭니다.
        hist = self.history.setdefault(track_id, self._new_track_state())

        # 이전 기억(프레임 좌표)을 꺼내고, 현재 프레임 좌표를 새로운 기억으로 덮어씁니다.
        prev_kps = hist["prev_kps"]
        prev_conf = hist["prev_conf"]
        hist["prev_kps"] = keypoints.copy()
        hist["prev_conf"] = kp_conf.copy()

        x1, y1, x2, y2 = box.astype(float)
        bh = max(y2 - y1, 1.0) # 카메라 거리에 따른 움직임 원근차를 보정하기 위해 사용하는 기준 값 (사람 키 크기)
        h, w = shape[:2]

        # 이전 프레임 기억이 없으면(방금 막 카메라에 찍힌 거라면) 움직임을 알 수 없으므로 0으로 시작합니다.
        if prev_kps is None or prev_conf is None:
            hist["motion_buf"].append(0.0)
            return 0.0, 0.0, 0.0

        def avg_disp(ids: List[int]) -> float:
            """현재 관절점(p1)과 과거 관절점(p0) 사이의 직선 거리(유클리드 거리)를 재는 내부 함수"""
            vals = []
            for i in ids:
                p1, p0 = keypoints[i], prev_kps[i]
                c1, c0 = kp_conf[i], prev_conf[i]
                # 과거와 현재 모두 유효한 좌표일 때만 측정합니다. (갑자기 튀는 오류 방지)
                if self._is_valid_kp(p1, c1, w, h) and self._is_valid_kp(p0, c0, w, h):
                    # 이동한 픽셀 거리를 사람의 픽셀 키(bh)로 나눠, 0.015 같은 '상대적 이동 비율'로 정규화!
                    vals.append(float(np.linalg.norm(p1 - p0) / bh))
            return float(np.mean(vals)) if vals else 0.0

        # 상체(팔 포함) 움직임의 평균값과 몸통 코어(어깨/골반) 움직임 평균값을 따로 잽니다.
        upper = avg_disp(self.UPPER_IDS)
        core = avg_disp(self.CORE_IDS)

        # 시스템 전체를 대표할 움직임 값은 상체와 코어 중 '더 큰 폭'으로 요동친 값을 취합니다.
        smooth = max(upper, core)
        hist["motion_buf"].append(smooth)            # 버퍼(큐)에 일단 집어넣고
        smooth = float(np.mean(hist["motion_buf"])) if hist["motion_buf"] else 0.0 # N프레임치 평균으로 출렁임을 다잡습니다 (Smoothing)
        return smooth, upper, core

    def _classify_motion(self, smooth: float, upper: float, core: float) -> str:
        """계산해낸 수학적 이동량 수치를 바탕으로 움직임(Motion) 크기를 4단계 텍스트로 라벨링합니다."""
        
        # [LOCAL_ONLY - 부분 움직임] : 팔이나 무릎(상체)은 크게 허우적대는데, 몸통 코어(어깨/골반)가 딱 박혀서 안 움직일 때
        # => 어딘가에 깔렸거나 모서리에 끼여서 빠져나오지 못하고 발버둥 치는 안타까운 상황의 주요 힌트가 됩니다.
        if upper >= self.cfg.motion_local_only_upper and core <= self.cfg.motion_local_only_core:
            return "LOCAL_ONLY"
            
        # [ACTIVE - 활발] 전체 움직임 버퍼 평균이나 순간 상체 움직임이 기준을 넘는 원활한 움직임
        if smooth >= self.cfg.motion_active_smooth or upper >= self.cfg.motion_active_upper:
            return "ACTIVE"
            
        # [LOW - 미세 움직임] 활발 수치엔 못 미치지만, 숨을 쉬거나 몸을 떠는 등 작은 수치라도 센싱될 때 (사망 판정 지연용)
        if smooth >= self.cfg.motion_low:
            return "LOW"
            
        # [NONE - 기절/사망] 로봇 카메라 노이즈 수준 이하의 어떤 유의미한 움직임도 없는 서늘한 상태
        return "NONE"

    # ------------------------------------------------------------------
    # 4) 보조 판정
    # ------------------------------------------------------------------
    @staticmethod
    def _possible_trapped(visibility: str, posture: str, motion: str) -> bool:
        """부분 노출/무움직임 등을 바탕으로 매몰 의심 여부를 계산한다."""
        if visibility == "PARTIAL" and motion in ("LOCAL_ONLY", "NONE"):
            return True
        if visibility == "UPPER_BODY" and posture == "COLLAPSED" and motion == "NONE":
            return True
        return False

    def _state_duration(self, track_id: int, signature: str) -> Tuple[float, float]:
        """전체 관측 시간과 현재 상태 지속 시간을 계산한다.

        - seen_sec : 이 사람을 처음 본 이후 경과 시간
        - state_sec: 현재 (visibility/posture/motion/trapped) 조합이 유지된 시간
        """
        hist = self.history.setdefault(track_id, self._new_track_state())
        now = time.time()
        if hist["last_signature"] != signature:
            hist["last_signature"] = signature
            hist["state_since"] = now
        return now - hist["first_seen"], now - hist["state_since"]

    # ------------------------------------------------------------------
    # 5) 최종 emergency_level 결정
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 5) 최종 위급 단계(Emergency Level) 결정: 모든 정보를 종합해 최종 판단을 내립니다!
    # ------------------------------------------------------------------
    def _decide(
        self,
        visibility: str,
        posture: str,
        motion: str,
        trapped: bool,
        seen_sec: float,
        state_sec: float,
    ) -> str:
        """관측 상태, 자세, 움직임, 그리고 '해당 상태가 얼마나 오랫동안 지속되었는가(state_sec)'를 
        종합하여 최종적으로 화면에 띄울 위급 단계(NORMAL~CRITICAL)를 결정합니다.

        [시간 지연(Time Delay) 로직의 이유]
        사람이 잠깐 신발끈을 묶으려고 허리를 숙이거나 실수로 넘어졌다가 바로 일어나는 경우에도 
        즉시 알람이 울리면 양치기 소년 시스템이 됩니다.
        따라서 특정 위험 자세(예: 누움, 움직임 없음)가 설정된 시간(예: 4.5초, 7초) 이상 
        **꾸준히 유지(지속)**될 때만 위급 단계를 서서히 올립니다.
        """
        # [ANALYZING] 카메라에 사람이 처음 잡힌 직후(기본 1.5초 이내)에는 
        # 움직임 버퍼에 데이터가 덜 쌓여서 오판할 확률이 높으므로 섣불리 판단하지 않고 대기합니다.
        if seen_sec < self.cfg.analyzing_sec:
            return "ANALYZING"

        # ==========================================
        # 조건 분기 1: 전신이 다 보일 때 (가장 확실한 판단 가능)
        # ==========================================
        if visibility == "FULL_BODY":
            # [CRITICAL] 누웠거나 붕괴된 최악의 자세 + 미동조차 없음 + 이게 7초(critical_sec) 이상 지속됨
            # => 심장마비, 완전 기절 등 즉시 출동해야 하는 초위급 상황
            if posture in ("LYING", "COLLAPSED") and motion == "NONE" and state_sec >= self.cfg.critical_sec:
                return "CRITICAL"
                
            # [WARNING] 쓰러졌는데 아주 미세하게 떨고 있거나, 아직 5.5초(warning_sec)밖에 안 지난 경우
            if posture in ("LYING", "COLLAPSED") and motion in ("LOW", "NONE") and state_sec >= self.cfg.warning_sec:
                return "WARNING"
                
            # [CAUTION] 다음 3가지 중 하나에 해당하면 주의(노란불)를 줍니다.
            # 1) 쓰러지진 않았지만 몸이 불안정하게 기운(Leaning) 상태로 4.5초(caution_sec) 이상 지속될 때
            # 2) 자세는 정상(Normal)인데, 5.5초 이상 꼼짝도 안 하고 서있거나 앉아있을 때 (졸도 직전 의심)
            # 3) 움직임이 활발하지 못하고 미세한 상태(LOW)가 4.5초 이상 지속될 때
            if (
                (posture == "LEANING" and motion in ("LOW", "NONE") and state_sec >= self.cfg.caution_sec)
                or (posture == "NORMAL" and motion == "NONE" and state_sec >= self.cfg.warning_sec)
                or (motion == "LOW" and state_sec >= self.cfg.caution_sec)
            ):
                return "CAUTION"
                
            # 위 모든 위험 조건을 피했다면 안전한 상태
            return "NORMAL"

        # ==========================================
        # 조건 분기 2: 상반신만 보일 때 (책상 앞, 벽 뒤)
        # ==========================================
        if visibility == "UPPER_BODY":
            # 상반신만 보이는데 엎드렸거나 고개가 완전히 꺾였고(Collapsed) 움직임도 없다면 경고
            if posture == "COLLAPSED" and motion == "NONE" and state_sec >= self.cfg.warning_sec:
                return "WARNING"
            # 엎드리진 않았지만 활발한 움직임이 4.5초 이상 아예 없다면 주의 (졸거나 쓰러진 상태 의심)
            if motion in ("LOW", "NONE") and state_sec >= self.cfg.caution_sec:
                return "CAUTION"
            return "NORMAL"

        # ==========================================
        # 조건 분기 3: 극히 일부만 보일 때 (잔해물 밑에 깔린 경우 등)
        # ==========================================
        if visibility == "PARTIAL":
            # 팔다리만 허우적대거나 꼼짝 안 하면서 '깔림(Trapped)'이 의심되는 상태로 오래 지속되면
            # 제대로 안 보임에도 불구하고 과감하게 구조 경고를 띄웁니다.
            if trapped and motion == "NONE" and state_sec >= self.cfg.critical_sec:
                return "WARNING"
            if motion in ("LOW", "NONE") and state_sec >= self.cfg.caution_sec:
                return "CAUTION"
            return "NORMAL"

        # 코드가 여기까지 도달할 일은 거의 없지만, 알 수 없는 예외 상황이라면 
        # 안전을 위해 무시하지 않고 일단 CAUTION을 띄워 로봇 관리자가 한 번 쳐다보게 만듭니다. (Fail-safe)
        return "CAUTION"

    # ------------------------------------------------------------------
    # 6) 시각화: 원본 카메라 프레임 위에 AI 분석 결과를 예쁘게 덧그립니다.
    # ------------------------------------------------------------------
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        kp_conf: np.ndarray,
        visibility: str,
        color: Tuple[int, int, int],
    ) -> None:
        """현재 판단된 가시성(visibility) 상태에 맞춰 카메라 프레임에 뼈대(Skeleton)를 그립니다.

        [시각화 규칙]
        - FULL_BODY(전신) 모드: 머리끝부터 발끝까지 17개 관절과 모든 뼈대를 연결해 그립니다.
        - 그 외(UPPER_BODY 등): 화면에 하체가 잘려서 안 보이는 상태이므로 상반신 뼈대만 그립니다.
        """
        if not self.cfg.draw_skeleton:
            return  # 설정에서 뼈대 그리기를 껐다면 그냥 넘어갑니다.

        h, w = frame.shape[:2]
        
        # 가시성에 따라 그릴 뼈대 선(links)과 관절 점(draw_ids)의 범위를 결정합니다.
        links = self.FULL_LINKS if visibility == "FULL_BODY" else self.UPPER_LINKS
        draw_ids = set(self.UPPER_IDS + self.LOWER_IDS) if visibility == "FULL_BODY" else set(self.UPPER_IDS)

        # 1. 뼈대(선) 그리기
        for a, b in links:
            pa, pb = keypoints[a], keypoints[b]
            ca, cb = kp_conf[a], kp_conf[b]
            # 이어질 두 관절이 모두 신뢰할 수 있는 화면 안의 좌표일 때만 선을 긋습니다.
            if self._is_valid_kp(pa, ca, w, h) and self._is_valid_kp(pb, cb, w, h):
                cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), color, 2)

        # 2. 관절(점) 그리기
        for i, p in enumerate(keypoints):
            if i not in draw_ids:
                continue # 상반신 모드인데 하체 번호면 스킵
            if not self._is_valid_kp(p, kp_conf[i], w, h):
                continue # 신뢰할 수 없는 튀는 좌표면 스킵
                
            center = tuple(p.astype(int))
            # 가독성을 위해 하얀 점(테두리 느낌)을 깔고 그 위에 색상 점을 덧칠합니다.
            cv2.circle(frame, center, 4, (255, 255, 255), -1)
            cv2.circle(frame, center, 3, color, -1)

    @staticmethod
    def _pack_result(
        track_id: int,
        bbox: Tuple[int, int, int, int],
        visibility: str,
        posture: str,
        motion: str,
        emergency_level: str,
        trapped: bool,
        seen_sec: float,
        state_sec: float,
        shoulder_tilt: float,
        head_drop_ratio: float,
        torso_angle: float,
        motion_smooth: float,
        motion_upper: float,
        motion_core: float,
        rep_point_px: Optional[List[int]],
        rep_point_method: str,
        shoulder_center_px: Optional[List[int]],
        hip_center_px: Optional[List[int]],
        face_anchor_px: Optional[List[int]],
    ) -> dict:
        """노드/대시보드에서 쓰기 쉬운 결과 dict를 생성한다."""
        return {
            "track_id": int(track_id),
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "observation": visibility,
            "posture": posture,
            "motion": motion,
            "emergency_level": emergency_level,
            "trapped": bool(trapped),
            "seen_sec": round(float(seen_sec), 3),
            "state_sec": round(float(state_sec), 3),
            "shoulder_tilt": round(float(shoulder_tilt), 3),
            "head_drop_ratio": round(float(head_drop_ratio), 3),
            "torso_angle": round(float(torso_angle), 3),
            "motion_smooth": round(float(motion_smooth), 5),
            "motion_upper": round(float(motion_upper), 5),
            "motion_core": round(float(motion_core), 5),
        }

    # ------------------------------------------------------------------
    # 메인 API: 외부(ROS2 노드)에서 실제로 호출하는 핵심 진입점(Entry Point) 함수들
    # ------------------------------------------------------------------
    def analyze_frame_with_results(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """카메라 프레임 1장을 통째로 분석해 '그림이 그려진 새 프레임'과 '분석 데이터 결과'를 동시에 반환합니다.

        [동작 흐름]
        1) YOLO Pose 모델에 이미지를 던져서 인간의 박스와 관절을 추론합니다.
        2) 화면에 잡힌 모든 사람(track_id)을 한 명씩 순회하면서:
            - 가시성 판정 (_classify_visibility)
            - 자세 판정 (_classify_posture)
            - 움직임 계산 및 분류 (_motion_value, _classify_motion)
            - 지속 시간을 재고 최종 위급 단계 결정 (_decide)
        3) 처리된 결과를 원본 텐서 이미지 위에 예쁘게 그리고(putText, rectangle),
        4) 웹 서버나 다른 ROS 노드에 텍스트로 던져주기 좋게 딕셔너리로 패킹합니다.
        """
        # 이미지에 덧그리기를 해야 하므로 원본을 훼손하지 않게 깊은 복사(copy)를 합니다.
        annotated = frame.copy()
        results: List[dict] = []

        # YOLO 모델 추론 (persist=True: 프레임 간 동일 인물 추적용 Track ID 부여 유지)
        yolo_results = self.model.track(frame, persist=True, verbose=False, conf=self.cfg.det_conf)
        
        # 사람이 아무도 안 잡혔거나 모델 오작동으로 데이터가 비었으면 바로 빈 결과를 리턴합니다.
        if not yolo_results or yolo_results[0].boxes is None or yolo_results[0].keypoints is None:
            return annotated, results

        # GPU(CUDA) 텐서로 잡혀있는 좌표 배열들을 파이썬에서 쓰기 좋게 CPU numpy 배열로 변환합니다.
        boxes_xyxy = yolo_results[0].boxes.xyxy.cpu().numpy()
        keypoints_xy = yolo_results[0].keypoints.xy.cpu().numpy()
        
        # 모델 종류에 따라 관절 확신도(confidence)가 없는 경우가 있으므로 예외처리(없으면 전부 1.0으로 강제 세팅)
        keypoints_conf = (
            yolo_results[0].keypoints.conf.cpu().numpy()
            if yolo_results[0].keypoints.conf is not None
            else np.ones((len(keypoints_xy), keypoints_xy.shape[1]), dtype=np.float32)
        )

        ids = yolo_results[0].boxes.id
        # 추적 ID(track id)가 있으면 쓰고 없으면 그냥 배열 순서대로 임시 ID(0, 1, 2...)를 붙여줍니다.
        track_ids = ids.int().cpu().tolist() if ids is not None else list(range(len(boxes_xyxy)))

        # 화면에 잡힌 각 사람마다 빙글빙글 돌면서 상세 분석을 시작합니다.
        for track_id, box, kps, kp_conf in zip(track_ids, boxes_xyxy, keypoints_xy, keypoints_conf):
            x1, y1, x2, y2 = box.astype(int)
            
            # 박스 좌표가 화면 밖을 뚫고 나가는 음수 에러를 막기 위한 화이트박스 방어 코드 (Clipping)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue # 면적이 0인 잘못된 박스면 스킵

            clipped_box = np.array([x1, y1, x2, y2], dtype=np.float32)

            # 1단계: 몸이 전신인지 상반신인지 가시성 판별
            visibility = self._classify_visibility(kps, kp_conf, frame.shape)

            # 2단계: 어깨나 허리 각도를 재서 현재 자세(서있음/쓰러짐 등) 판별
            posture, shoulder_tilt, head_drop_ratio, torso_angle = self._classify_posture(
                kps, kp_conf, clipped_box, visibility, frame.shape
            )

            # 3단계: 이전 프레임 좌표와 비교해서 흔들림(움직임) 크기 판별
            smooth, upper, core = self._motion_value(track_id, kps, kp_conf, clipped_box, frame.shape)
            motion = self._classify_motion(smooth, upper, core)

            # 4단계: 잔해물 등에 깔린 상태인지, 이 위험한 자세로 몇 초나 누워있었는지 시간 측정
            trapped = self._possible_trapped(visibility, posture, motion)
            signature = f"{visibility}|{posture}|{motion}|{trapped}"
            seen_sec, state_sec = self._state_duration(track_id, signature)
            
            # 5단계: 대망의 최종 위급 등급 산정 (WARNING? CRITICAL?)
            emergency_level = self._decide(visibility, posture, motion, trapped, seen_sec, state_sec)

            # 5-1) 대표 포인트(Rep. Point) 추출 (v0.620 관제용 기능)
            rep_point, shoulder_center, hip_center, face_anchor, rep_method = self._extract_rep_points(
                kps, kp_conf, clipped_box, visibility, frame.shape
            )

            # 6단계: 모든 상태 분석이 끝났으니, 그 결과에 맞춰 사람 몸 위에 색깔과 점을 덧칠합니다.
            color = self.COLORS[emergency_level] # 위급 단계별 색상 지정 (NORMAL=초록, CRITICAL=빨강 등)
            self._draw_skeleton(annotated, kps, kp_conf, visibility, color)

            # 대표 포인트(Rep. Point) 시각화: 십자선(+) 표시
            if rep_point is not None:
                rx, ry = int(rep_point[0]), int(rep_point[1])
                cv2.drawMarker(annotated, (rx, ry), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

            # 설정에서 그리기 옵션이 켜져있다면 사람 주변을 네모 박스로 감쌉니다.
            if self.cfg.draw_box:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 머리 위에 띄워줄 디버깅 및 정보 텍스트 조합
            line1 = f"ID {track_id} | {emergency_level}"
            line2 = f"{visibility} | {posture} | {motion}"
            line3 = f"tilt:{shoulder_tilt:.1f} hds:{head_drop_ratio:.2f} m:{smooth:.3f}"

            ty = max(20, y1 - 10) # 글씨가 캔버스 천장을 뚫지 않게 Y좌표 방어
            # 위급도 텍스트(line1)는 항상 크게 출력
            cv2.putText(annotated, line1, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            # 개발자/관제용 디버깅 텍스트(line2, line3)는 설정이 켜져있을 때만 발 밑에 출력
            if self.cfg.show_debug:
                cv2.putText(
                    annotated,
                    line2,
                    (x1, min(frame.shape[0] - 25, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    line3,
                    (x1, min(frame.shape[0] - 5, y2 + 42)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # 6) structured result 생성
            results.append(
                self._pack_result(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    visibility=visibility,
                    posture=posture,
                    motion=motion,
                    emergency_level=emergency_level,
                    trapped=trapped,
                    seen_sec=seen_sec,
                    state_sec=state_sec,
                    shoulder_tilt=shoulder_tilt,
                    head_drop_ratio=head_drop_ratio,
                    torso_angle=torso_angle,
                    motion_smooth=smooth,
                    motion_upper=upper,
                    motion_core=core,
                    rep_point_px=self._pt_to_list(rep_point),
                    rep_point_method=rep_method,
                    shoulder_center_px=self._pt_to_list(shoulder_center),
                    hip_center_px=self._pt_to_list(hip_center),
                    face_anchor_px=self._pt_to_list(face_anchor),
                )
            )

        return annotated, results

    def analyze_frame(self, frame: np.ndarray) -> np.ndarray:
        """annotated frame만 필요한 경우 사용하는 단순 API."""
        annotated, _ = self.analyze_frame_with_results(frame)
        return annotated

    @staticmethod
    def results_to_json(results: List[dict]) -> str:
        """결과 리스트를 JSON 문자열로 변환한다.

        ROS2 String topic, 로깅, 디버깅 용도로 사용 가능하다.
        """
        return json.dumps({"detections": results}, ensure_ascii=False)

    def extract_frame_emergency_level(self, results_list: List[dict]) -> Optional[str]:
        """
        프레임 단위 최종 emergency_level 하나를 만든다.
        현재 시나리오는 한 화면 1명 전제지만, 혹시 여러 개가 잡혀도
        가장 높은 severity를 대표값으로 사용한다.

        사람이 없으면 None 반환.
        """
        if not results_list:
            return None

        return max(
            (r["emergency_level"] for r in results_list),
            key=lambda x: self.EMERGENCY_PRIORITY.get(x, -1),
        )

    def analyze_frame_with_emergency_level(self, frame: np.ndarray):
        """
        ROS2 노드에서 쓰기 쉽게
        1) 시각화 프레임
        2) 프레임 대표 emergency_level 하나
        를 반환한다.
        """
        annotated, results_list = self.analyze_frame_with_results(frame)
        emergency_level = self.extract_frame_emergency_level(results_list)
        return annotated, emergency_level

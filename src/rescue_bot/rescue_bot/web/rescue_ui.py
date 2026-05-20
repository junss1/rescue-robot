'''
파일명: rescue_ui.py
날짜: 2026-03-16
버전: v1.100

버전 변경사항
- v1.200: collapse를 제외한 History 저장 항목에 당시 robot6/image_result 스트림 스냅샷을 파일로 보관하고 조회 카드에서 썸네일로 표시하도록 확장함
- v1.100: collapse를 제외한 History 저장 항목에 robot6 amcl_pose를 함께 적재하고 alerts 조회 응답에서 카드 하단 표시용 pose 필드를 분리해 제공함
- v1.006: direct MJPEG 스트림 stale 상태를 조회하는 API를 추가해 프론트에서 마지막 프레임을 강제로 숨길 수 있도록 보강함
- v1.005: direct MJPEG 스트림이 일정 시간 stale 상태면 응답을 종료해 UI가 마지막 프레임 대신 placeholder로 복귀하도록 조정함
- v1.004: CAM 03 direct stream 구독 토픽을 실제 overlay 출력인 /output/cam01/compressed 로 복구함
- v1.003: CAM 03 direct stream 토픽을 /output/cam01/compressed 에서 /output/cam01 로 조정함
- v1.002: UI alerts 조회 응답에서 사용하지 않는 emotion 필드를 제거함
- v1.001: /robot6/session/result를 DB 전용으로 적재하고 빈 값은 null로 정규화하며, UI alerts 조회 목록에서는 제외함
- v1.000: camera_system 최종 출력 토픽명을 /output/cam01/compressed, /output/cam02/compressed 로 정렬하고 Flask direct stream relay도 새 토픽을 구독하도록 조정함
- v0.901: direct MJPEG 재시도 중 잘못 붙은 /stream/cam03&t=... 경로도 수용하도록 스트림 키 정규화를 추가함
- v0.900: camera_system의 /output/cam01, /output/cam02 CompressedImage를 Flask 직접 MJPEG 스트리밍으로 중계해 CAM 03, CAM 04 표시를 web_video_server 의존 없이 지원함
- v0.800: collapse 감지를 Flask API(/api/record_collapse) 저장 방식으로 변경하고, 기존 TTS/pose와 동일한 DB 적재 경로로 통일함
- v0.700: collapse alert ROS 토픽(/alert/cam01/collapse, /alert/cam02/collapse)을 백엔드에서 직접 구독해 SQLite/History 연동과 UI 팝업 폴링 기반 표시를 지원함
- v0.600: victim_pose_stamped 히스토리 저장 API를 추가하고, History 패널에 요구조자 좌표 기록을 포함할 수 있도록 확장함
- v0.500: UI용 API(/api/system_status, /api/robot_state, /api/alerts, /api/map_summary)를 추가하고, 향후 ROS/Gazebo 실제 연동 전환이 쉽도록 데이터 구성 함수를 분리함
- v0.400: rescue_ui를 메인 Flask 서버로 고정하고, ROS 웹 연동 설정(rosbridge/map/alert/stream)을 템플릿 컨텍스트로 분리해 welcome 템플릿 기준으로 통합함
- v0.300: UI/day5/login 라우트는 유지하면서, SQLite 연동 준비를 위한 서비스 함수 분리 및 설정 상수를 추가함
- v0.200: 로그인 페이지, 세션 인증, 로그아웃, 인증 보호 데코레이터를 추가해 대시보드 접근을 로그인 기반으로 변경함
- v0.110: day5 형태에 맞게 Flask 서버를 render_template 기반으로 리팩토링하고 대시보드 UI를 templates로 분리함
- v0.020: 앱 팩토리(create_app), 설정 상수, health check 라우트를 추가해 실행 구조를 정리함
- v0.001: resetMapView 호출 대비 템플릿 연동과 기본 실행부를 안정화함
'''

import hmac
import json
import os
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from functools import wraps
from typing import Any, Dict, Optional

from flask import Flask, Response, flash, redirect, render_template, request, session, stream_with_context, url_for

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
except ImportError:
    rclpy = None
    SingleThreadedExecutor = None
    Node = object
    QoSProfile = None
    ReliabilityPolicy = None
    CompressedImage = None

APP_HOST = os.getenv('SRD_FLASK_HOST', '0.0.0.0')
APP_PORT = int(os.getenv('SRD_FLASK_PORT', '5000'))
APP_DEBUG = os.getenv('SRD_FLASK_DEBUG', 'false').lower() == 'true'
APP_SECRET_KEY = os.getenv('SRD_SECRET_KEY', 'srd-day5-dev-key')
DASHBOARD_VERSION = 'v1.0'
LOGIN_USERNAME = os.getenv('SRD_LOGIN_USERNAME', 'admin')
LOGIN_PASSWORD = os.getenv('SRD_LOGIN_PASSWORD', '1234')
DEFAULT_DASHBOARD_TITLE = 'RR 통합 관제 시스템'
SQLITE_DB_PATH = os.getenv('SRD_SQLITE_PATH', 'srd_mission_records.db')
ROSBRIDGE_WS_DEFAULT = os.getenv('SRD_ROSBRIDGE_WS_URL', 'ws://{host}:9090')
ROS_MAP_TOPIC = os.getenv('SRD_ROS_MAP_TOPIC', '/robot5/map')
ROS_ALERT_TOPIC = os.getenv('SRD_ROS_ALERT_TOPIC', '/srd/severity_data')
ROS_POSE_TOPIC = os.getenv('SRD_ROS_POSE_TOPIC', '/robot6/amcl_pose')
ROS_PLAN_TOPIC = os.getenv('SRD_ROS_PLAN_TOPIC', '/robot6/plan')
MJPEG_ENDPOINT = os.getenv('SRD_MJPEG_ENDPOINT', 'http://{host}:8080/stream')
# CAM 01 임시 기본 토픽 (원복: /robot6/image_result/compressed -> /robot6/image_result)
MJPEG_TOPIC = os.getenv('SRD_MJPEG_TOPIC', '/robot6/image_result')
MJPEG_WIDTH = int(os.getenv('SRD_MJPEG_WIDTH', '640'))
MJPEG_HEIGHT = int(os.getenv('SRD_MJPEG_HEIGHT', '480'))
ALERTS_QUERY_LIMIT = int(os.getenv('SRD_ALERTS_QUERY_LIMIT', '50'))
COLLAPSE_ALERT_LABELS = {
    'cam01': 'CAM 03',
    'cam02': 'CAM 04',
}
OUTPUT_STREAM_TOPICS = {
    'cam03': '/output/cam01/compressed',
    'cam04': '/output/cam02/compressed',
}
OUTPUT_STREAM_STALE_TIMEOUT_SEC = float(os.getenv('SRD_OUTPUT_STREAM_STALE_TIMEOUT_SEC', '2.0'))
OUTPUT_STREAM_STATE = {
    key: {'frame': None, 'seq': 0, 'updated_at': 0.0}
    for key in OUTPUT_STREAM_TOPICS
}
OUTPUT_STREAM_CONDITIONS = {
    key: threading.Condition()
    for key in OUTPUT_STREAM_TOPICS
}
OUTPUT_STREAM_STARTED = False
OUTPUT_STREAM_LOCK = threading.Lock()
HISTORY_SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'history_snapshots')


def get_login_credentials() -> Dict[str, str]:
    return {
        'username': LOGIN_USERNAME,
        'password': LOGIN_PASSWORD,
    }


def validate_login(username: str, password: str) -> bool:
    credentials = get_login_credentials()
    is_valid_user = hmac.compare_digest(username, credentials['username'])
    is_valid_password = hmac.compare_digest(password, credentials['password'])
    return is_valid_user and is_valid_password


def build_base_template_context() -> Dict[str, Any]:
    return {
        'dashboard_title': DEFAULT_DASHBOARD_TITLE,
        'dashboard_version': DASHBOARD_VERSION,
    }


def build_dashboard_context(username: Optional[str]) -> Dict[str, Any]:
    context = build_base_template_context()
    context['username'] = username or 'operator'
    context['rosbridge_ws_default'] = ROSBRIDGE_WS_DEFAULT
    context['ros_map_topic'] = ROS_MAP_TOPIC
    context['ros_alert_topic'] = ROS_ALERT_TOPIC
    context['ros_pose_topic'] = ROS_POSE_TOPIC
    context['ros_plan_topic'] = ROS_PLAN_TOPIC
    context['mjpeg_endpoint'] = MJPEG_ENDPOINT
    context['mjpeg_topic'] = MJPEG_TOPIC
    context['mjpeg_width'] = MJPEG_WIDTH
    context['mjpeg_height'] = MJPEG_HEIGHT
    return context


def get_sqlite_config() -> Dict[str, str]:
    """
    SQLite 연동 준비용 설정 제공 함수.
    실제 DB 연결 로직은 추후 이 함수/전용 모듈을 통해 확장한다.
    """
    return {
        'db_path': SQLITE_DB_PATH,
    }


def get_sqlite_connection(read_only: bool = True):
    """
    SQLite 연결 생성.
    - read_only=True: 읽기 전용 (mode=ro)
    - read_only=False: 쓰기 가능
    - 우선순위 1: SRD_SQLITE_PATH
    - 우선순위 2: 현재 작업 경로의 srd_mission_records.db
    - 우선순위 3: 패키지 내 database/data/srd_mission_records.db
    """
    candidate_paths = []

    configured_path = SQLITE_DB_PATH.strip()
    if configured_path:
        if os.path.isabs(configured_path):
            candidate_paths.append(configured_path)
        else:
            candidate_paths.append(os.path.abspath(configured_path))

    package_db_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'database',
            'data',
            'srd_mission_records.db',
        )
    )
    if package_db_path not in candidate_paths:
        candidate_paths.append(package_db_path)

    for db_path in candidate_paths:
        if os.path.exists(db_path):
            uri = f'file:{db_path}?mode=ro' if read_only else f'file:{db_path}'
            return sqlite3.connect(uri, uri=True, check_same_thread=False)

    return None


def ensure_history_snapshot_dir() -> None:
    os.makedirs(HISTORY_SNAPSHOT_DIR, exist_ok=True)


def save_history_snapshot(image_url: Optional[str]) -> Optional[str]:
    if not image_url:
        return None

    parsed = urllib.parse.urlparse(str(image_url))
    if parsed.scheme not in ('http', 'https'):
        return None

    ensure_history_snapshot_dir()
    filename = f"history_{int(time.time() * 1000)}.jpg"
    filepath = os.path.join(HISTORY_SNAPSHOT_DIR, filename)

    request_headers = {
        'User-Agent': 'rescue-ui-history-snapshot/1.0',
        'Accept': 'image/*',
    }
    request_obj = urllib.request.Request(image_url, headers=request_headers)

    try:
        with urllib.request.urlopen(request_obj, timeout=3) as response:
            content = response.read()
    except Exception:
        return None

    if not content:
        return None

    with open(filepath, 'wb') as snapshot_file:
        snapshot_file.write(content)

    return f'/static/history_snapshots/{filename}'


def build_history_status_payload(
    status_msg: str,
    robot6_amcl_pose: Optional[Dict[str, Any]] = None,
    robot6_image_snapshot_url: Optional[str] = None,
) -> str:
    if not robot6_amcl_pose and not robot6_image_snapshot_url:
        return status_msg

    payload = {
        'message': status_msg,
        'robot6_amcl_pose': None,
        'robot6_image_snapshot_url': robot6_image_snapshot_url,
    }

    if robot6_amcl_pose:
        payload['robot6_amcl_pose'] = {
            'x': robot6_amcl_pose.get('x'),
            'y': robot6_amcl_pose.get('y'),
            'yaw': robot6_amcl_pose.get('yaw'),
        }

    return json.dumps(payload, ensure_ascii=False)


def parse_history_status_payload(raw_status_msg: Any) -> Dict[str, Any]:
    if not isinstance(raw_status_msg, str):
        return {
            'message': raw_status_msg,
            'robot6_amcl_pose': None,
            'robot6_image_snapshot_url': None,
        }

    try:
        parsed = json.loads(raw_status_msg)
    except (TypeError, ValueError):
        return {
            'message': raw_status_msg,
            'robot6_amcl_pose': None,
            'robot6_image_snapshot_url': None,
        }

    if not isinstance(parsed, dict):
        return {
            'message': raw_status_msg,
            'robot6_amcl_pose': None,
            'robot6_image_snapshot_url': None,
        }

    robot6_amcl_pose = parsed.get('robot6_amcl_pose')
    if not isinstance(robot6_amcl_pose, dict):
        robot6_amcl_pose = None

    robot6_image_snapshot_url = parsed.get('robot6_image_snapshot_url')
    if not isinstance(robot6_image_snapshot_url, str) or not robot6_image_snapshot_url.strip():
        robot6_image_snapshot_url = None

    return {
        'message': parsed.get('message', raw_status_msg),
        'robot6_amcl_pose': robot6_amcl_pose,
        'robot6_image_snapshot_url': robot6_image_snapshot_url,
    }


def insert_severity_log(
    track_id: str,
    severity: str,
    status_msg: str,
    robot6_amcl_pose: Optional[Dict[str, Any]] = None,
    image_url: Optional[str] = None,
) -> None:
    conn = get_sqlite_connection(read_only=False)
    if conn is None:
        raise RuntimeError('DB connection failed')

    try:
        cursor = conn.cursor()
        snapshot_url = save_history_snapshot(image_url)
        cursor.execute(
            '''
            INSERT INTO severity_logs (track_id, severity, status_msg, timestamp)
            VALUES (?, ?, ?, datetime('now', 'localtime'))
            ''',
            (track_id, severity, build_history_status_payload(status_msg, robot6_amcl_pose, snapshot_url)),
        )
        conn.commit()
    finally:
        conn.close()


def normalize_session_result_value(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {
            key: normalize_session_result_value(item)
            for key, item in value.items()
        }
        return normalized or None
    if isinstance(value, list):
        normalized = [normalize_session_result_value(item) for item in value]
        return normalized or None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


class OutputStreamRelay(Node):
    def __init__(self):
        super().__init__('output_stream_relay')
        self.frame_subscriptions = []

        image_qos = QoSProfile(depth=1)
        image_qos.reliability = ReliabilityPolicy.RELIABLE

        for stream_key, topic_name in OUTPUT_STREAM_TOPICS.items():
            subscription = self.create_subscription(
                CompressedImage,
                topic_name,
                lambda msg, key=stream_key: self._image_callback(msg, key),
                image_qos,
            )
            self.frame_subscriptions.append(subscription)

        self.get_logger().info('Output stream relay started')

    def _image_callback(self, msg: CompressedImage, stream_key: str) -> None:
        condition = OUTPUT_STREAM_CONDITIONS[stream_key]
        with condition:
            OUTPUT_STREAM_STATE[stream_key]['frame'] = bytes(msg.data)
            OUTPUT_STREAM_STATE[stream_key]['seq'] += 1
            OUTPUT_STREAM_STATE[stream_key]['updated_at'] = time.monotonic()
            condition.notify_all()


def start_output_stream_relay() -> None:
    global OUTPUT_STREAM_STARTED

    if (
        rclpy is None
        or CompressedImage is None
        or SingleThreadedExecutor is None
        or QoSProfile is None
        or ReliabilityPolicy is None
    ):
        return

    with OUTPUT_STREAM_LOCK:
        if OUTPUT_STREAM_STARTED:
            return
        OUTPUT_STREAM_STARTED = True

    def worker() -> None:
        try:
            rclpy.init(args=None)
        except RuntimeError:
            pass

        executor = SingleThreadedExecutor()
        node = OutputStreamRelay()
        executor.add_node(node)

        try:
            executor.spin()
        finally:
            executor.remove_node(node)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    threading.Thread(target=worker, name='output-stream-relay', daemon=True).start()


def generate_output_stream(stream_key: str):
    last_seq = 0
    condition = OUTPUT_STREAM_CONDITIONS[stream_key]

    while True:
        with condition:
            condition.wait_for(
                lambda: OUTPUT_STREAM_STATE[stream_key]['seq'] != last_seq,
                timeout=1.0,
            )
            frame = OUTPUT_STREAM_STATE[stream_key]['frame']
            seq = OUTPUT_STREAM_STATE[stream_key]['seq']
            updated_at = OUTPUT_STREAM_STATE[stream_key]['updated_at']

        if not frame:
            continue

        if seq == last_seq and updated_at > 0.0:
            if (time.monotonic() - updated_at) >= OUTPUT_STREAM_STALE_TIMEOUT_SEC:
                with condition:
                    OUTPUT_STREAM_STATE[stream_key]['frame'] = None
                return

        last_seq = seq

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n'
            b'Cache-Control: no-cache\r\n\r\n' + frame + b'\r\n'
        )


def get_output_stream_status(stream_key: str) -> Dict[str, Any]:
    state = OUTPUT_STREAM_STATE[stream_key]
    updated_at = float(state.get('updated_at', 0.0) or 0.0)
    age_sec = max(0.0, time.monotonic() - updated_at) if updated_at > 0.0 else None
    has_frame = bool(state.get('frame'))
    is_stale = (not has_frame) or (age_sec is not None and age_sec >= OUTPUT_STREAM_STALE_TIMEOUT_SEC)

    return {
        'stream_key': stream_key,
        'topic': OUTPUT_STREAM_TOPICS[stream_key],
        'has_frame': has_frame,
        'seq': int(state.get('seq', 0)),
        'updated_at': updated_at if updated_at > 0.0 else None,
        'age_sec': age_sec,
        'stale_timeout_sec': OUTPUT_STREAM_STALE_TIMEOUT_SEC,
        'is_stale': is_stale,
    }


def build_empty_alerts_data() -> Dict[str, Any]:
    return {
        'count': 0,
        'items': [],
        'source': 'unavailable',
    }


def build_health_payload() -> Dict[str, Any]:
    sqlite_config = get_sqlite_config()
    sqlite_conn = get_sqlite_connection()
    sqlite_ready = sqlite_conn is not None
    if sqlite_conn is not None:
        sqlite_conn.close()
    return {
        'status': 'ok',
        'service': 'srd-flask-server',
        'version': DASHBOARD_VERSION,
        'login_enabled': True,
        'sqlite_ready': sqlite_ready,
        'sqlite_db_path': sqlite_config['db_path'],
    }


def get_system_status_data() -> Dict[str, Any]:
    health = build_health_payload()
    return {
        'system': health,
        'ros': {
            'bridge_ws': ROSBRIDGE_WS_DEFAULT,
            'map_topic': ROS_MAP_TOPIC,
            'alert_topic': ROS_ALERT_TOPIC,
            'connected': None,
        },
        'stream': {
            'endpoint_template': MJPEG_ENDPOINT,
            'topic': MJPEG_TOPIC,
            'width': MJPEG_WIDTH,
            'height': MJPEG_HEIGHT,
            'connected': None,
        },
    }


def get_robot_state_data() -> Dict[str, Any]:
    return {
        'robot_id': None,
        'mode': 'unavailable',
        'connected': None,
        'battery_percent': None,
        'position': {
            'x': None,
            'y': None,
            'yaw': None,
        },
        'source': 'unavailable',
    }


def get_alerts_data() -> Dict[str, Any]:
    conn = None
    try:
        conn = get_sqlite_connection()
        if conn is None:
            return build_empty_alerts_data()

        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            '''
            SELECT COUNT(*) AS total_count
            FROM severity_logs
            WHERE track_id != 'SESSION_RESULT'
            '''
        )
        total_count_row = cursor.fetchone()
        total_count = int(total_count_row['total_count']) if total_count_row else 0

        cursor.execute(
            '''
            SELECT
                id,
                timestamp,
                track_id,
                severity,
                status_msg,
                motion_score,
                is_lying
            FROM severity_logs
            WHERE track_id != 'SESSION_RESULT'
            ORDER BY id DESC
            LIMIT ?
            ''',
            (ALERTS_QUERY_LIMIT,),
        )

        items = []
        for row in cursor.fetchall():
            parsed_status = parse_history_status_payload(row['status_msg'])
            items.append(
                {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'track_id': row['track_id'],
                    'severity': row['severity'],
                    'status_msg': parsed_status['message'],
                    'robot6_amcl_pose': parsed_status['robot6_amcl_pose'],
                    'robot6_image_snapshot_url': parsed_status['robot6_image_snapshot_url'],
                    'motion_score': row['motion_score'],
                    'is_lying': bool(row['is_lying']) if row['is_lying'] is not None else None,
                }
            )

        return {
            'count': total_count,
            'items': items,
            'source': 'sqlite',
        }
    except Exception:
        return build_empty_alerts_data()
    finally:
        if conn is not None:
            conn.close()


def get_map_summary_data() -> Dict[str, Any]:
    return {
        'topic': ROS_MAP_TOPIC,
        'available': None,
        'resolution': None,
        'width': None,
        'height': None,
        'source': 'unavailable',
    }


def is_authenticated() -> bool:
    return bool(session.get('is_authenticated'))


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not is_authenticated():
            flash('먼저 로그인해 주세요.', 'warning')
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)

    return wrapped_view


def create_app() -> Flask:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, 'templates'),
    )
    app.config['SECRET_KEY'] = APP_SECRET_KEY
    start_output_stream_relay()

    @app.route('/')
    def home():
        if is_authenticated():
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if is_authenticated():
            return redirect(url_for('dashboard'))

        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')

            if validate_login(username, password):
                session.clear()
                session['is_authenticated'] = True
                session['username'] = username
                flash('로그인되었습니다.', 'success')
                return redirect(url_for('dashboard'))

            flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'danger')
            return redirect(url_for('login'))

        return render_template(
            'login_center_srd.html',
            **build_base_template_context(),
        )

    @app.route('/dashboard')
    @login_required
    def dashboard():
        return render_template(
            'welcome_center_srd.html',
            **build_dashboard_context(session.get('username')),
        )

    @app.route('/logout')
    def logout():
        session.clear()
        flash('로그아웃되었습니다.', 'info')
        return redirect(url_for('login'))

    @app.route('/health')
    def health():
        return build_health_payload()

    @app.route('/api/system_status')
    @login_required
    def api_system_status():
        return get_system_status_data()

    @app.route('/api/robot_state')
    @login_required
    def api_robot_state():
        return get_robot_state_data()

    @app.route('/api/alerts')
    @login_required
    def api_alerts():
        return get_alerts_data()

    @app.route('/api/map_summary')
    @login_required
    def api_map_summary():
        return get_map_summary_data()

    @app.route('/stream/<stream_key>')
    @login_required
    def stream_output(stream_key: str):
        normalized_key = stream_key.split('&', 1)[0]
        if normalized_key not in OUTPUT_STREAM_TOPICS:
            return {'status': 'error', 'message': 'invalid stream_key'}, 404

        return Response(
            stream_with_context(generate_output_stream(normalized_key)),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )

    @app.route('/api/stream_status/<stream_key>')
    @login_required
    def api_stream_status(stream_key: str):
        normalized_key = stream_key.split('&', 1)[0]
        if normalized_key not in OUTPUT_STREAM_TOPICS:
            return {'status': 'error', 'message': 'invalid stream_key'}, 404
        return get_output_stream_status(normalized_key)

    @app.route('/api/record_tts', methods=['POST'])
    @login_required
    def api_record_tts():
        data = request.json
        try:
            insert_severity_log(
                f"TTS_{data.get('type')}",
                'INFO',
                data.get('message'),
                data.get('robot6_amcl_pose'),
                data.get('image_url'),
            )
            return {'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500

    @app.route('/api/record_victim_pose', methods=['POST'])
    @login_required
    def api_record_victim_pose():
        data = request.json or {}
        x = data.get('x')
        y = data.get('y')
        z = data.get('z')
        status_msg = f"Victim pose received: X={x}, Y={y}, Z={z}"

        try:
            insert_severity_log(
                'VICTIM_POSE',
                'INFO',
                status_msg,
                data.get('robot6_amcl_pose'),
                data.get('image_url'),
            )
            return {'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500

    @app.route('/api/record_collapse', methods=['POST'])
    @login_required
    def api_record_collapse():
        data = request.json or {}
        cam_key = str(data.get('cam_key', '')).strip().lower()
        label = COLLAPSE_ALERT_LABELS.get(cam_key)
        if label is None:
            return {'status': 'error', 'message': 'invalid cam_key'}, 400

        try:
            insert_severity_log(
                f'COLLAPSE_{cam_key.upper()}',
                'CRITICAL',
                f'{label} 붕괴 감지!',
            )
            return {'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500

    @app.route('/api/record_session_result', methods=['POST'])
    @login_required
    def api_record_session_result():
        data = request.json or {}
        raw_result = data.get('result')

        if isinstance(raw_result, str):
            try:
                raw_result = json.loads(raw_result)
            except json.JSONDecodeError:
                raw_result = {'raw': raw_result}

        normalized_result = normalize_session_result_value(raw_result)

        try:
            insert_severity_log(
                'SESSION_RESULT',
                'INFO',
                json.dumps(normalized_result, ensure_ascii=False),
                data.get('robot6_amcl_pose'),
                data.get('image_url'),
            )
            return {'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500

    return app


app = create_app()


def main():
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG, use_reloader=False)


if __name__ == '__main__':
    main()

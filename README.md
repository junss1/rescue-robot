# 다중 로봇 기반 재난 구조 탐색·관제 시스템

[![Demo Video](docs/rescue-robot.png)](https://youtu.be/MMzPYvMR2ZU)
↑ 이미지 클릭 시 데모 영상을 확인할 수 있습니다.

재난 상황 시나리오를 가정한 다중 로봇 기반 구조 시스템입니다.

카메라 감지 시스템, 탐색 로봇의 사람 검출 이벤트, 구조 로봇의 이동/비전 분석/음성 안내/웹 관제를 하나의 ROS 2 워크스페이스에서 통합 실행할 수 있도록 구성했습니다.

이 프로젝트는 카메라 영상 기반 객체 감지와 탐색 이벤트를 이용해 구조 로봇의 후속 행동을 자동화합니다. Robot5가 탐색 중 피해자를 검출하면, Robot6가 해당 위치 기반 구조 미션을 수행하고 비전 분석 및 음성 상호작용을 진행합니다.

---

## 1. 프로젝트 개요

* 분야: 재난 구조 로봇 / 실내 탐색 / 피해자 감지 / 구조 관제
* 주요 기술: ROS 2 Humble, Python, OpenCV, YOLO, Flask, STT/TTS
* 기준 환경: Ubuntu 22.04, ROS 2 Humble, Python 3.10
* 주요 구성:

  * 다중 카메라 감지 노드
  * 탐색 로봇 사람 검출 패키지
  * 구조 미션 제어 노드
  * Nav2 기반 이동 제어
  * 피해자 분석 및 음성 안내 노드
  * 웹 기반 구조 관제 UI

---

## 2. 프로젝트 목표

* 카메라 영상에서 사람과 붕괴 상황을 감지합니다.

- 탐색 로봇이 실내 탐색 중 피해자 검출 이벤트를 생성합니다.
- 구조 로봇이 검출 위치를 기반으로 자동 이동합니다.
- 도착 후 비전 분석 및 음성 기반 구조 절차를 수행합니다.
- 웹 UI를 통해 구조 상황과 ROS 데이터를 모니터링합니다.

---

## 3. 저장소 구조

```text
rescue-robot-workspace/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── src/
    ├── camera_system/
    │   ├── camera_system/
    │   ├── launch/
    │   ├── models/
    │   ├── package.xml
    │   └── setup.py
    │
    ├── robot5_person_search/
    │   ├── robot5_person_search/
    │   ├── launch/
    │   ├── package.xml
    │   └── setup.py
    │
    └── rescue_bot/
        ├── rescue_bot/
        │   ├── analyzer/
        │   ├── models/
        │   └── web/
        ├── docs/
        ├── launch/
        ├── package.xml
        └── setup.py
```

### 폴더 설명

| 경로                               | 설명                  |
| -------------------------------- | ------------------- |
| `src/camera_system/`             | 카메라 감지 및 붕괴 감지 패키지  |
| `src/robot5_person_search/`      | 탐색 중 사람 검출 이벤트 패키지  |
| `src/rescue_bot/`                | 구조 미션 제어 및 웹 UI 패키지 |
| `src/rescue_bot/rescue_bot/web/` | Flask 기반 웹 관제 UI    |
| `src/rescue_bot/docs/`           | 런타임 계약 및 문서         |

---

## 4. 외부 의존성

이 저장소에는 ROS 2 시스템 패키지를 포함하지 않습니다.

아래 패키지는 외부 환경에 별도로 설치해야 합니다.

```bash
rosdep install --from-paths src --ignore-src -r -y
```

추가적으로 다음 패키지 구성이 필요합니다.

```text
Nav2
TurtleBot4 packages
rosbridge_server
web_video_server
cv_bridge
tf2_ros
nav2_simple_commander
```

---

## 5. 주요 패키지

### 5-1. `camera_system`

YOLO 기반 객체 감지 및 붕괴 감지를 수행하는 패키지입니다.

| 실행 이름               | 역할                   |
| ------------------- | -------------------- |
| `camera_publisher`  | USB 카메라 영상 발행        |
| `detection_node`    | YOLO 기반 사람 감지        |
| `overlay_node`      | Detection Overlay 생성 |
| `collapse_detector` | 붕괴 감지 이벤트 생성         |

주요 출력:

```text
/output/cam01
/output/cam02
/detection/cam01/person
/detection/cam02/person
/alert/cam01/collapse
/alert/cam02/collapse
```

### 5-2. `robot5_person_search`

Robot5 탐색 중 피해자 검출 이벤트를 생성하는 패키지입니다.

| 실행 이름                       | 역할             |
| --------------------------- | -------------- |
| `person_event_detector`     | 피해자 검출 이벤트 생성  |
| `explore_detect_supervisor` | 탐색 상태 및 이벤트 관리 |

주요 역할:

* 실내 탐색 중 사람 감지
* 피해자 방향/위치 이벤트 생성
* 구조 로봇 연동 이벤트 발행

### 5-3. `rescue_bot`

Robot6 구조 로봇 미션을 제어하는 패키지입니다.

| 실행 이름                 | 역할            |
| --------------------- | ------------- |
| `rescue_control_node` | 구조 미션 흐름 제어   |
| `rescue_nav_node`     | Nav2 기반 이동 제어 |
| `rescue_stt_node`     | 음성 인식 처리      |
| `rescue_ui`           | 웹 기반 구조 관제 UI |

주요 역할:

* 구조 위치 이동
* 도착 이벤트 처리
* 피해자 상태 분석
* TTS/STT 기반 구조 대화
* 웹 UI 기반 관제

---

## 6. 시스템 FLOW

```text
camera_system
    ↓
robot5_person_search
    ↓
victim detection event
    ↓
rescue_bot navigation
    ↓
arrival event
    ↓
vision analysis
    ↓
TTS / STT interaction
    ↓
next mission or docking
```

---

## 7. Technical Highlights

### 7-1. Multi-Robot Event Pipeline

탐색 로봇과 구조 로봇을 이벤트 기반으로 연결했습니다.

```text
Robot5 Detection
→ Victim Event
→ Robot6 Dispatch
→ Arrival Event
→ Rescue Interaction
```

탐색과 구조 로직을 분리하여 확장성과 유지보수성을 높였습니다.

### 7-2. Vision / Navigation / Voice 분리

구조 시스템을 기능 계층 기준으로 분리했습니다.

```text
Vision Layer
→ Navigation Layer
→ Rescue Interaction Layer
```

각 노드는 ROS 2 토픽 및 이벤트 기반으로 연결됩니다.

### 7-3. Web 기반 구조 관제

Flask 기반 웹 UI를 통해 다음 정보를 확인할 수 있습니다.

* 구조 이벤트 상태
* 카메라 스트림
* ROS 데이터
* 미션 진행 상태

### 7-4. Victim Interaction Flow

구조 로봇은 도착 후 음성 상호작용을 수행합니다.

```text
Arrival
→ Victim Analysis
→ TTS Request
→ STT Response
→ Next Action
```

---

## 8. 주요 토픽

### Detection / Event

| 토픽                        | 타입              | 설명         |
| ------------------------- | --------------- | ---------- |
| `/detection/cam01/person` | Detection Event | 사람 검출 이벤트  |
| `/alert/cam01/collapse`   | Bool/Event      | 붕괴 감지 이벤트  |
| `/robot5/person_event`    | Event           | 피해자 검출 이벤트 |

### Rescue Mission

| 토픽                        | 타입     | 설명       |
| ------------------------- | ------ | -------- |
| `/robot6/mission/arrived` | Bool   | 구조 위치 도착 |
| `/robot6/tts/request`     | String | 음성 안내 요청 |
| `/robot6/tts/done`        | Bool   | 음성 출력 완료 |

---

## 9. 설치 및 빌드

### 9-1. ROS 2 환경 로드

```bash
source /opt/ros/humble/setup.bash
```

### 9-2. 워크스페이스 생성

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 9-3. 프로젝트 클론

```bash
git clone <repository-url> rescue-robot-workspace
```

### 9-4. Python 의존성 설치

```bash
cd rescue-robot-workspace
pip install -r requirements.txt
```

### 9-5. ROS 의존성 설치

```bash
rosdep install --from-paths src --ignore-src -r -y
```

### 9-6. 빌드

```bash
colcon build --symlink-install
source install/setup.bash
```

---

## 10. 실행 방법

### 10-1. camera_system 실행

```bash
ros2 launch camera_system camera_system.launch.py
```

### 10-2. robot5_person_search 실행

```bash
ros2 launch robot5_person_search robot5_person_search.launch.py
```

### 10-3. rescue_bot 실로봇 런타임 실행

```bash
ros2 launch rescue_bot rescue_real.launch.py
```

### 10-4. rescue_bot 웹 UI 실행

```bash
ros2 launch rescue_bot rescue_web.launch.py
```

---

## 11. 모델 파일

```text
src/camera_system/models/best26.pt
src/camera_system/models/yolov8n-seg.pt
src/robot5_person_search/yolo11n.pt
src/rescue_bot/rescue_bot/models/yolo11n-pose.pt
```

---

## 12. Hardware Configuration

| 항목             | 내용                   |
| -------------- | -------------------- |
| OS             | Ubuntu 22.04         |
| ROS            | ROS 2 Humble         |
| Navigation     | Nav2                 |
| Robot Platform | TurtleBot4 기반        |
| Vision         | YOLO                 |
| Web UI         | Flask                |
| Audio          | STT / TTS            |
| 의사소통  | ROS 2 Topic/Event 기반 |

---

## 13. 주의사항

* Nav2 및 TurtleBot4 관련 패키지는 외부 환경에 설치해야 합니다.
* 실제 로봇 실행 전 카메라 장치, 오디오 장치, 네임스페이스 설정을 환경에 맞게 수정해야 합니다.
* 웹 UI 실행 시 `rosbridge_server`, `web_video_server` 설치가 필요합니다.
* 공개 환경에서는 Flask 로그인 정보를 환경변수 기반으로 관리하는 것을 권장합니다.

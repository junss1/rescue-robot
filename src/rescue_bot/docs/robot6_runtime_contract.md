# Robot6 Runtime Contract 2026-03-17

실로봇 기준 현재 런타임 계약을 코드 기준으로 정리한 문서입니다.

## Core Flow
- 입력 시작점은 `rescue_nav_node`의 `/robot5/robot_pose_at_detection` + `/robot5/victim_point` 입니다.
- `rescue_nav_node`가 목표 주행 성공 시 `/robot6/mission/arrived` 를 발행합니다.
- `robot6_control_node`가 `arrived`를 받아 세션을 시작하고 비전 분석을 진행합니다.
- `robot6_control_node`는 분석 결과에 따라 `/robot6/tts/request` 에 상태 문자열을 발행합니다.
- `rescue_dialogue_node`는 상태 문자열을 처리하고 종료 시 `/robot6/tts/done=True` 를 발행합니다.
- `rescue_nav_node`는 `tts/done` 수신 후 다음 목표 또는 도킹 단계로 진행합니다.

## Topic Contract
- `/robot5/robot_pose_at_detection`
  - type: `geometry_msgs/PoseStamped`
  - role: nav의 주 목표 입력
- `/robot5/victim_point`
  - type: `geometry_msgs/PointStamped`
  - role: nav 목표 yaw 계산 보조 입력
- `/robot6/mission/arrived`
  - type: `std_msgs/Bool`
  - producer: nav
  - consumer: control
- `/robot6/tts/request`
  - type: `std_msgs/String`
  - producer: control
  - consumer: stt
  - payload: `NORMAL`, `CAUTION`, `WARNING`, `CRITICAL`
- `/robot6/tts/done`
  - type: `std_msgs/Bool`
  - producer: stt
  - consumers: control, nav
- `/rescue/victim_pose_stamped`
  - type: `geometry_msgs/PoseStamped`
  - producer: control
  - role: UI/로그/후속 확장용 출력
  - note: 현재 nav의 직접 입력이 아님

## Runtime Dependencies
- control: `cv_bridge`, `opencv`, `numpy`, `ultralytics`
- nav: `turtlebot4_navigation`
- stt: `pygame`, `speech_recognition`, `torch`, `whisper`, `gtts`, 마이크, 오디오 출력 장치

## Launch Baseline
- 기준 런치는 `launch/rescue_real.launch.py`
- 세 노드는 모두 `use_sim_time=False` 기준으로 실행합니다.

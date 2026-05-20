# rescue_real.launch.py v0.100 2026-03-17
# [이번 버전에서 수정된 사항]
# - 실로봇용 rescue_bot 노드(control/nav/stt) 실행 런치 파일 추가
# - 실로봇 기준 런타임 계약(robot5 입력, arrived/control/stt 연계) 설명 보강

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Rescue Robot 6 실로봇용 런치 파일.
    - rescue_control_node: 도착 이후 세션 시작, 비전 분석, 상태/결과/TTS 요청 발행
    - rescue_nav_node: /robot5/robot_pose_at_detection + /robot5/victim_point 기반 목표 주행
    - rescue_stt_node: /robot6/tts/request 상태 문자열을 받아 대화 시나리오 수행 후 done 발행

    기준 연동 흐름:
    robot5 입력 -> rescue_nav_node -> /robot6/mission/arrived
    -> robot6_control_node 세션 시작/분석
    -> /robot6/tts/request
    -> rescue_dialogue_node
    -> /robot6/tts/done
    -> rescue_nav_node 다음 목표 또는 도킹
    """

    control_node = Node(
        package='rescue_bot',
        executable='rescue_control_node',
        name='robot6_control_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    nav_node = Node(
        package='rescue_bot',
        executable='rescue_nav_node',
        name='rescue_nav_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    stt_node = Node(
        package='rescue_bot',
        executable='rescue_stt_node',
        name='rescue_dialogue_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    return LaunchDescription([
        control_node,
        nav_node,
        stt_node,
    ])

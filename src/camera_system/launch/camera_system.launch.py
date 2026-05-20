#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """카메라 시스템 통합 런칭 설정"""

    return LaunchDescription([
        # 카메라 퍼블리셔 노드
        Node(
            package='camera_system',
            executable='camera_publisher',
            name='camera_publisher',
            output='screen',
        ),

        # YOLO 객체 감지 노드
        Node(
            package='camera_system',
            executable='detection_node',
            name='detection_node',
            output='screen',
        ),

        # 오버레이 시각화 노드
        Node(
            package='camera_system',
            executable='overlay_node',
            name='overlay_node',
            output='screen',
        ),

        # 붕괴 감지 노드
        Node(
            package='camera_system',
            executable='collapse_detector',
            name='collapse_detector',
            output='screen',
        ),
    ])

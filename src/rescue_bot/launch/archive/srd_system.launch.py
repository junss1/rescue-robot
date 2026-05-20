import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    분석 노드(Analyzer)와 데이터베이스 노드(Database)를 동시에 실행하는 런치 파일
    """
    # 1. SRD 통합 분석 노드 실행 (얼굴, 포즈, 위급도 판단)
    analyzer_node = Node(
        package='rescue_bot',
        executable='analyzer',
        name='srd_advanced_analyzer',
        output='screen'
    )

    # 2. SRD 데이터베이스 저장 노드 실행 (기록 및 로깅)
    database_node = Node(
        package='rescue_bot',
        executable='database',
        name='srd_database_node',
        output='screen'
    )
    
    return LaunchDescription([
        analyzer_node,
        database_node
    ])
# rescue_web.launch.py v0.102 2026-03-15
# [이번 버전에서 수정된 사항]
# - web UI, rosbridge, web_video_server를 함께 실행하는 웹 전용 런치 파일 추가
# - 선택적으로 브라우저를 자동 실행하는 옵션 추가
# - rescue_ui를 console script 대신 Python 모듈 직접 실행으로 변경해 설치 메타데이터 의존성 제거
# - 기본 실행 시 브라우저가 자동으로 열리도록 open_browser 기본값과 browser_url 기본값을 조정

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    flask_host = LaunchConfiguration('flask_host')
    flask_port = LaunchConfiguration('flask_port')
    rosbridge_port = LaunchConfiguration('rosbridge_port')
    web_video_port = LaunchConfiguration('web_video_port')
    open_browser = LaunchConfiguration('open_browser')
    browser_url = LaunchConfiguration('browser_url')

    arguments = [
        DeclareLaunchArgument(
            'flask_host',
            default_value='0.0.0.0',
            description='Flask UI bind host',
        ),
        DeclareLaunchArgument(
            'flask_port',
            default_value='5000',
            description='Flask UI port',
        ),
        DeclareLaunchArgument(
            'rosbridge_port',
            default_value='9090',
            description='rosbridge websocket port',
        ),
        DeclareLaunchArgument(
            'web_video_port',
            default_value='8080',
            description='web_video_server port',
        ),
        DeclareLaunchArgument(
            'open_browser',
            default_value='true',
            description='Whether to open the UI URL in a local browser',
        ),
        DeclareLaunchArgument(
            'browser_url',
            default_value=['http://127.0.0.1:', flask_port],
            description='URL to open when open_browser is true',
        ),
    ]

    ui_env = [
        SetEnvironmentVariable('SRD_FLASK_HOST', flask_host),
        SetEnvironmentVariable('SRD_FLASK_PORT', flask_port),
        SetEnvironmentVariable('SRD_ROSBRIDGE_WS_URL', ['ws://{host}:', rosbridge_port]),
        SetEnvironmentVariable('SRD_MJPEG_ENDPOINT', ['http://{host}:', web_video_port, '/stream']),
    ]

    ui_node = ExecuteProcess(
        cmd=['python3', '-m', 'rescue_bot.web.rescue_ui'],
        name='rescue_ui',
        output='screen',
    )

    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
        parameters=[{
            'port': rosbridge_port,
        }],
    )

    web_video_node = Node(
        package='web_video_server',
        executable='web_video_server',
        name='web_video_server',
        output='screen',
        parameters=[{
            'port': web_video_port,
        }],
    )

    open_ui_browser = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=['xdg-open', browser_url],
                output='screen',
            ),
        ],
        condition=IfCondition(open_browser),
    )

    return LaunchDescription(
        arguments + ui_env + [
            rosbridge_node,
            web_video_node,
            ui_node,
            open_ui_browser,
        ]
    )

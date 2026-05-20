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

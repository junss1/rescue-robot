from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir


def generate_launch_description():
    """
    호환용 런치 파일.
    기존 rescue_system.launch.py 호출은 rescue_real.launch.py로 연결한다.
    """

    real_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/rescue_real.launch.py'])
    )

    return LaunchDescription([
        real_launch,
    ])

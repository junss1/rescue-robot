from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot5_person_search',
            executable='person_event_detector',
            name='person_event_detector',
            namespace='robot5',
            output='screen',
            remappings=[
                ('/tf', '/robot5/tf'),
                ('/tf_static', '/robot5/tf_static'),
            ],
        ),
        Node(
            package='robot5_person_search',
            executable='explore_detect_supervisor',
            name='explore_detect_supervisor',
            namespace='robot5',
            output='screen',
            remappings=[
                ('/tf', '/robot5/tf'),
                ('/tf_static', '/robot5/tf_static'),
            ],
        ),
    ])
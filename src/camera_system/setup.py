import os
from glob import glob

from setuptools import setup

package_name = 'camera_system'

models_files = [os.path.relpath(path) for path in glob('models/*.pt')]

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/camera_system.launch.py']),
        ('share/' + package_name + '/models', models_files),
    ],
    install_requires=[
        'setuptools',
        'opencv-python>=4.5.0',
        'numpy>=1.20.0',
        'ultralytics>=8.0.0',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'pillow>=8.0.0',
    ],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='maintainer@example.com',
    description='ROS 2 camera system with YOLO detection and collapse detection.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = camera_system.camera_publisher:main',
            'detection_node = camera_system.detection_node:main',
            'overlay_node = camera_system.overlay_node:main',
            'collapse_detector = camera_system.collapse_detector:main',
        ],
    },
)

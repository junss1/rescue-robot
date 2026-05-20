# setup.py v0.001 2026-03-15
# [이번 버전에서 수정된 사항]
# - colcon symlink build 충돌을 일으키던 package_dir 매핑 제거
# - 불필요한 setuptools import 정리

import os
from glob import glob

from setuptools import setup

package_name = 'camera_system'

# models 디렉토리의 모든 .pt 파일 수집
models_files = []
models_dir = os.path.join(os.path.dirname(__file__), 'models')
if os.path.exists(models_dir):
    models_files = glob(os.path.join(models_dir, '*.pt'))
    models_files = [os.path.relpath(f) for f in models_files]

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/camera_system.launch.py']),
        ('share/' + package_name + '/models', models_files),
    ],
    install_requires=['setuptools', 'opencv-python>=4.5.0', 'numpy>=1.20.0', 'ultralytics>=8.0.0', 'torch>=1.9.0', 'torchvision>=0.10.0', 'pillow>=8.0.0'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@example.com',
    description='ROS2 Camera System with YOLO Detection and Collapse Detection',
    license='Apache License 2.0',
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

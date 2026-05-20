# v0.111
# file: setup.py
# date: 2026-03-17
# changes:
# - robot5 연동 테스트용 new nav 별도 entry point 추가
# - rescue_nav_node_sucees entry point 추가
# - robot6_control_node entry point 추가
# - rescue_stt_node entry point 추가
# - analyzer 정리 기준에 맞춰 archive 이동 파일 엔트리 포인트 제거
# - 실로봇 기준 엔트리 포인트 계약(control/nav/stt) 설명 주석 보강

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rescue_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chans',
    maintainer_email='ahwkt46@gmail.com',
    description='Rescue robot orchestration package for Robot6 mission control, navigation, STT/TTS, and web UI.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # Real runtime: arrived -> control session -> tts request -> stt done -> nav next goal
            'rescue_control_node = rescue_bot.analyzer.rescue_control_node:main',
            'rescue_nav_node = rescue_bot.analyzer.rescue_nav_node:main',
            'rescue_stt_node = rescue_bot.analyzer.rescue_stt_node:main',
            'rescue_ui = rescue_bot.web.rescue_ui:main',
        ],
    },
)

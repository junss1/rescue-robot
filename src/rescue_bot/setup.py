import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'rescue_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('rescue_bot/models/*.pt')),
        (os.path.join('share', package_name, 'web', 'templates'), glob('rescue_bot/web/templates/*.html')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='maintainer@example.com',
    description='Rescue robot orchestration package for Robot6 mission control, navigation, STT/TTS, and web UI.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'rescue_control_node = rescue_bot.analyzer.rescue_control_node:main',
            'rescue_nav_node = rescue_bot.analyzer.rescue_nav_node:main',
            'rescue_stt_node = rescue_bot.analyzer.rescue_stt_node:main',
            'rescue_ui = rescue_bot.web.rescue_ui:main',
        ],
    },
)

from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot5_person_search'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='hea0919@naver.com',
    description='Robot5 person search package for exploration supervision and victim detection event publishing.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'person_event_detector = robot5_person_search.person_event_detector:main',
            'explore_detect_supervisor = robot5_person_search.explore_detect_supervisor:main',
        ],
    },
)

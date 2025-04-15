from setuptools import setup
import os
from glob import glob

package_name = 'robotControl'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[

        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        (os.path.join('share',package_name,'launch'),
	 glob(os.path.join('launch','*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chacrica Pagadala',
    maintainer_email='pagadc@rpi.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_processor = robotControl.nodes.image_processor:main',
            'robot_controller = robotControl.nodes.robot_controller:main',
        ],
    },
)

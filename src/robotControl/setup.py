from setuptools import setup

package_name = 'robotControl'

setup(
    name=package_name,
    packages=[package_name],
    data_files=[
        # This ensures the package is registered with ROS2
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # This installs the package.xml
        ('share/' + package_name, ['package.xml']),
        # Include launch files in the installation
        ('share/' + package_name + '/launch', ['launch/robotControl.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Robot control using colored signal detection.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_processor = robotControl.nodes.image_processor:main',
            'robot_controller = robotControl.nodes.robot_controller:main',
        ],
    },
)

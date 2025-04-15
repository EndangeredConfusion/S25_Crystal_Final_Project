import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robotControl',
            executable='image_processor.py',
            name='image_processor'
        ),
        Node(
            package='robotControl',
            executable='robot_controller.py',
            name='robot_controller'
        )
    ])

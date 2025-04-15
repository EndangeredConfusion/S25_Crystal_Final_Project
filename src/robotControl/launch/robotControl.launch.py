import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    image_processor = Node(

    #########
            package = 'robotControl',
    ########
            
            executable = 'image_processor')
    robot_controller = Node(

    #########
            package = 'robotControl',
    ########
            
            executable = 'robot_controller')



    return LaunchDescription([
        image_processor,
        robot_controller])

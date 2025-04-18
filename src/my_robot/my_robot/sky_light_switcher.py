import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import subprocess

class RedLightGreenLight(Node):
    def __init__(self):
        super().__init__('light_switcher')
        self.current_color = 'green'
        self.timer = self.create_timer(5.0, self.switch_light)
        self.model_path = os.path.expanduser('~/S25_Crystal_Final_Project/src/my_robot/models')

    def switch_light(self):
        # Delete the current light (if it exists)
        command = f'ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "{{name: \\"{self.current_color}_light\\"}}"'
        subprocess.run(command, shell=True)

        # Switch color
        self.current_color = 'red' if self.current_color == 'green' else 'green'
        model_file = os.path.join(self.model_path, f'{self.current_color}_light', 'model.sdf')

        # Spawn the new light
        command = (
            f'ros2 run gazebo_ros spawn_entity.py '
            f'-file {model_file} '
            f'-entity {self.current_color}_light '
            f'-x 0 -y 0 -z 0.5'
        )

        subprocess.run(command, shell=True)

        if self.current_color == 'green':
            self.robot_mover.enable_movement()
        else:
            self.robot_mover.disable_movement()

class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.move_in_circle)  # 10 Hz
        self.green_light_active = False  # You toggle this based on light status
        self.angular_speed = 0.5
        self.linear_speed = 0.2

    def move_in_circle(self):
        if not self.green_light_active:
            return  # Stand still if not green

        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = self.angular_speed
        self.publisher.publish(twist)

    def stop(self):
        twist = Twist()  # all zeros
        self.publisher.publish(twist)

    def enable_movement(self):
        self.green_light_active = True

    def disable_movement(self):
        self.green_light_active = False
        self.stop()


def main():
    rclpy.init()
    node = RedLightGreenLight()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
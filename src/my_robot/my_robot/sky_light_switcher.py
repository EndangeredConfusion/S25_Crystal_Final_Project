import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import subprocess

class RedLightGreenLight(Node):
    def __init__(self):
        super().__init__('light_switcher')
        self.current_color = 'green'
        self.model_path = os.path.expanduser('~/S25_Crystal_Final_Project/src/my_robot/models')

        # Robot movement setup
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.green_light_active = True
        self.timer_movement = self.create_timer(0.1, self.move_in_circle)  # 10 Hz

        # Light switching setup
        self.timer_light = self.create_timer(5.0, self.switch_light)

    def switch_light(self):
        # Delete the current light
        command = f'ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "{{name: \\"{self.current_color}_light\\"}}"'
        subprocess.run(command, shell=True)

        # Switch color
        self.current_color = 'red' if self.current_color == 'green' else 'green'
        model_file = os.path.join(self.model_path, f'{self.current_color}_light', 'model.sdf')

        # Spawn new light
        command = (
            f'ros2 run gazebo_ros spawn_entity.py '
            f'-file {model_file} '
            f'-entity {self.current_color}_light '
            f'-x 0 -y 0 -z 0.5'
        )
        subprocess.run(command, shell=True)

        # Control movement based on color
        if self.current_color == 'green':
            self.green_light_active = True
        else:
            self.green_light_active = False
            self.stop_robot()

    def move_in_circle(self):
        if not self.green_light_active:
            return

        twist = Twist()
        twist.linear.x = 0.2
        twist.angular.z = 0.5
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()  # Zero velocity
        self.cmd_vel_pub.publish(twist)


def main():
    rclpy.init()
    node = RedLightGreenLight()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
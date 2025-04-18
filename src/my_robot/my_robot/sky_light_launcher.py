import os
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
import time
import subprocess

class RedLightGreenLight(Node):
    def __init__(self):
        super().__init__('light_switcher')
        self.current_color = 'green'
        self.timer = self.create_timer(5.0, self.switch_light)
        self.model_path = os.path.expanduser('~/ros2_ws/src/my_robot/models')

    def switch_light(self):
        # Delete the current light (if it exists)
        subprocess.run(['ros2', 'service', 'call', '/delete_entity', 'gazebo_msgs/srv/DeleteEntity',
                        f'{{name: "{self.current_color}_light"}}'], shell=True)
        time.sleep(0.5)

        # Switch color
        self.current_color = 'red' if self.current_color == 'green' else 'green'
        model_file = os.path.join(self.model_path, f'{self.current_color}_light', 'model.sdf')

        # Spawn the new light
        subprocess.run([
            'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
            '-file', model_file,
            '-entity', f'{self.current_color}_light',
            '-x', '0', '-y', '0', '-z', '3'
        ], shell=True)

def main():
    rclpy.init()
    node = RedLightGreenLight()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
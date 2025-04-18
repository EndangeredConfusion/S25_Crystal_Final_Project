#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue
from rcl_interfaces.msg import ParameterType
import time


class SkyLightSwitcher(Node):
    def __init__(self):
        super().__init__('sky_light_switcher')
        self.cli = self.create_client(SetParameters, '/gazebo/set_parameters')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_parameters service...')

        self.green = True
        self.timer = self.create_timer(5.0, self.toggle_sky)  # toggle every 5 seconds

    def toggle_sky(self):
        if self.green:
            self.set_sky_color(0.8, 0.0, 0.0)  # Red
            self.get_logger().info('🔴 RED LIGHT!')
        else:
            self.set_sky_color(0.0, 0.6, 0.0)  # Green
            self.get_logger().info('🟢 GREEN LIGHT!')
        self.green = not self.green

    def set_sky_color(self, r, g, b):
        req = SetParameters.Request()
        param = Parameter()
        param.name = 'background_color'
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_STRING,
            string_value=f"{r} {g} {b} 1.0"
        )
        req.parameters.append(param)
        future = self.cli.call_async(req)

        # Optionally wait a little (non-blocking)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = SkyLightSwitcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
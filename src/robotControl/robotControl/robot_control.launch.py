#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info("Robot controller node started!")
        self.subscription = self.create_subscription(
            String,
            'control_signal',
            self.command_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Define speeds for different commands (adjust as needed)
        self.full_speed = 0.5   # start moving
        self.slow_speed = 0.2   # slow down
        self.turn_rate = 0.5    # angular velocity for turning

    def command_callback(self, msg: String):
        command = msg.data.lower()
        twist = Twist()

        if command == "stop":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info("Received stop command.")
        elif command == "start":
            twist.linear.x = self.full_speed
            twist.angular.z = 0.0
            self.get_logger().info("Received start command.")
        elif command == "slow":
            twist.linear.x = self.slow_speed
            twist.angular.z = 0.0
            self.get_logger().info("Received slow command.")
        elif command == "turn_left":
            twist.linear.x = self.full_speed  # Keep moving forward
            twist.angular.z = self.turn_rate   # Positive value for left turn
            self.get_logger().info("Received turn_left command.")
        elif command == "turn_right":
            twist.linear.x = self.full_speed
            twist.angular.z = -self.turn_rate  # Negative value for right turn
            self.get_logger().info("Received turn_right command.")
        else:
            self.get_logger().warn(f"Unknown command received: {command}")
            return

        self.cmd_pub.publish(twist)
        self.get_logger().info(f"Published Twist command: {twist}")

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

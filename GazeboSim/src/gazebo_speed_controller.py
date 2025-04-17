import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist

class ColorSpeedController(Node):
    def __init__(self):
        super().__init__('color_speed_controller')
        self.subscription = self.create_subscription(
            Int32MultiArray,
            '/color_counts',
            self.color_counts_callback,
            10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Color indices CSS3 names
        self.green_idx = 63
        self.red_idx = 118
        self.yellow_idx = 139
        
        self.green_speed = 0.5
        self.yellow_speed = 0.2
        self.red_speed = 0.0
        
        self.get_logger().info("Speed controller ready!")

    def color_counts_callback(self, msg):
        twist = Twist()
        counts = msg.data
        
        # Safety check for array length
        if len(counts) < 140:
            self.get_logger().error("Invalid color counts array length")
            return

        # Priority: Red > Yellow > Green
        if counts[self.red_idx] > 0:
            twist.linear.x = self.red_speed
            self.get_logger().info("RED: Full stop")
        elif counts[self.yellow_idx] > 0:
            twist.linear.x = self.yellow_speed
            self.get_logger().info("YELLOW: Slowing down")
        elif counts[self.green_idx] > 0:
            twist.linear.x = self.green_speed
            self.get_logger().info("GREEN: Full speed ahead")
        else:
            twist.linear.x = 0.0  # Default stop
            self.get_logger().info("No color detected: Stopping")

        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    controller = ColorSpeedController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

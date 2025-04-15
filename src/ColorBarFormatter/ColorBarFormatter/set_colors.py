from Rosmaster_Lib import Rosmaster
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Int32MultiArray
import webcolors

web_color_names = webcolors.names()

bot = Rosmaster()

def color_callback(count_array: Int32MultiArray):
    color_counts = np.array(count_array.data)
    top_n_color_indices_descending = np.argsort(color_counts)[-4:][::-1]
    total_count = np.sum(color_counts[top_n_color_indices_descending])
    counts = np.floor(color_counts[top_n_color_indices_descending] / total_count * 16)
    remainder = 16 - np.sum(counts )
    counts[0] += remainder

    color_values = [webcolors.name_to_rgb(web_color_names[index]) for index in top_n_color_indices_descending]

    current_pos = 0
    for i in range(4):
        count = counts[i]
        current_color = color_values[i]
        r = current_color.red
        g = current_color.green
        b = current_color.blue

        for _ in range(count):
            bot.set_colorful_lamps(current_pos, r, g, b)
            current_pos += 1


class ColorSetter(Node):
    def __init__(self):
        super().__init__('set_colors')        
        self.get_logger().info("Color bar publisher!")
        self.create_subscription(Int32MultiArray, "/color_counts", color_callback, 10)


def main(args=None):
    rclpy.init(args=args)
    node = ColorSetter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

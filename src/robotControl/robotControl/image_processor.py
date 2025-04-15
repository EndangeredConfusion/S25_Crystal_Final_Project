import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2 as cv
import numpy as np
from sklearn.neighbors import KDTree
import torch
import webcolors
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Get CSS3 colors from webcolors and build a list of color names.
web_color_names = list(webcolors.CSS3_NAMES_TO_HEX.keys())
color_names = np.array(web_color_names)

# Build reference BGR colors.
reference_BGR_colors = {}
for color in web_color_names:
    rgb = webcolors.hex_to_rgb(webcolors.CSS3_NAMES_TO_HEX[color])
    # Convert RGB to BGR by reordering.
    reference_BGR_colors[color] = (rgb.blue, rgb.green, rgb.red)

# Precompute CIELAB values for each reference color.
precomputed_CIELAB_values = np.array([
    cv.cvtColor(np.uint8([[reference_BGR_colors[label]]]), cv.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    for label in color_names
])

# Build a KDTree for fast nearest-neighbor queries in LAB space.
color_tree = KDTree(precomputed_CIELAB_values)
color_values = np.array([reference_BGR_colors[label] for label in color_names])

def discretize_image_CPU(BGR_image):
    """Discretize the image on CPU using KDTree in LAB space."""
    lab_pixels = cv.cvtColor(BGR_image, cv.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab_pixels.shape[:2]
    lab_flat = lab_pixels.reshape(-1, 3)
    _, nearest_indices = color_tree.query(lab_flat, k=1)
    recolored_flat = color_values[nearest_indices.flatten()]
    output_image = recolored_flat.reshape(h, w, 3).astype(np.uint8)
    counts = np.bincount(nearest_indices.flatten(), minlength=len(color_names))
    return output_image, counts

def discretize_image_GPU(BGR_image):
    """Discretize the image using GPU acceleration with PyTorch if available."""
    # Enhance image contrast slightly.
    img = cv.convertScaleAbs(BGR_image, alpha=1.5, beta=0)
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    h, w = img_lab.shape[:2]
    img_lab_flat = torch.tensor(img_lab.reshape(-1, 3), device='cuda')
    lab_refs = torch.tensor(precomputed_CIELAB_values, device='cuda')
    diffs = img_lab_flat[:, None, :] - lab_refs[None, :, :]
    distances = torch.norm(diffs, dim=2)
    nearest_indices = torch.argmin(distances, dim=1)
    rgb_refs = torch.tensor(color_values, device='cuda')
    recolored_flat = rgb_refs[nearest_indices]
    recolored = recolored_flat.view(h, w, 3).cpu().numpy().astype(np.uint8)
    counts = torch.bincount(nearest_indices, minlength=len(color_names)).cpu().numpy()
    return recolored, counts

def detect_arrow_direction(BGR_image):
    """
    Detect a green arrow and return its direction as 'turn_left' or 'turn_right'.
    """
    # Convert to HSV for robust color segmentation.
    hsv = cv.cvtColor(BGR_image, cv.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    
    # Handle different versions of OpenCV for contour detection.
    contours_info = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    
    for cnt in contours:
        if cv.contourArea(cnt) < 1000:
            continue
        epsilon = 0.03 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        # A simplistic approach: if the approximated contour has 7 vertices, assume an arrow.
        if len(approx) == 7:
            M = cv.moments(cnt)
            if M['m00'] == 0:
                continue
            cX = int(M['m10'] / M['m00'])
            width = BGR_image.shape[1]
            return "turn_left" if cX < width // 2 else "turn_right"
    return None

def decide_command(BGR_image, counts):
    """
    Decide which command to send based on the fraction of red, yellow,
    or green pixels in the image.
    """
    total_pixels = BGR_image.shape[0] * BGR_image.shape[1]
    red_threshold = 0.15 * total_pixels
    green_threshold = 0.15 * total_pixels
    yellow_threshold = 0.15 * total_pixels

    try:
        red_index = np.where(color_names == "red")[0][0]
        green_index = np.where(color_names == "green")[0][0]
        yellow_index = np.where(color_names == "yellow")[0][0]
    except IndexError:
        return None

    if counts[red_index] > red_threshold:
        return "stop"
    elif counts[yellow_index] > yellow_threshold:
        return "slow"
    elif counts[green_index] > green_threshold:
        # If there is a significant green area, check for an arrow.
        arrow_cmd = detect_arrow_direction(BGR_image)
        return arrow_cmd if arrow_cmd is not None else "start"
    else:
        return None

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.get_logger().info("Image processor node started!")
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        # Publish commands on the "control_signal" topic.
        self.command_pub = self.create_publisher(String, 'control_signal', 10)
        self.bridge = CvBridge()
        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            self.get_logger().info("CUDA available, using GPU for image processing.")
        else:
            self.get_logger().info("CUDA not available, using CPU for image processing.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error("CvBridge error: " + str(e))
            return

        if self.CUDA:
            processed_img, counts = discretize_image_GPU(frame)
        else:
            processed_img, counts = discretize_image_CPU(frame)
        
        # Decide on a command based on the processed image.
        command = decide_command(frame, counts)
        if command:
            msg_out = String()
            msg_out.data = command
            self.command_pub.publish(msg_out)
            self.get_logger().info(f"Published command: {command}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

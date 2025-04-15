import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2 as cv
import numpy as np
import os
from sklearn.neighbors import KDTree
import torch
import webcolors


web_color_names = webcolors.names()  
color_names = np.array(web_color_names)

reference_BGR_colors = {color_name: webcolors.name_to_rgb(color_name) for color_name in web_color_names}
reference_BGR_colors = {k: (v[2], v[1], v[0]) for k, v in reference_BGR_colors.items()}

precomputed_CIELAB_values = np.array([
    cv.cvtColor(np.uint8([[reference_BGR_colors[label]]]), cv.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    for label in color_names
])

color_tree = KDTree(precomputed_CIELAB_values)
color_values = np.array([reference_BGR_colors[label] for label in color_names])

def descritize_image_CPU(BGR_image):
    lab_pixels = cv.cvtColor(BGR_image, cv.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab_pixels.shape[:2]
    lab_flat = lab_pixels.reshape(-1, 3)
    _, nearest_indices = color_tree.query(lab_flat, k=1)
    recolored_flat = color_values[nearest_indices.flatten()]
    output_image = recolored_flat.reshape(h, w, 3).astype(np.uint8)
    counts = np.bincount(nearest_indices.flatten(), minlength=len(color_names))
    return output_image, counts

def descritize_image_GPU(BGR_image):
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
    Returns 'turn_left', 'turn_right', or None.
    """
    # Convert to HSV to help isolate green regions
    hsv = cv.cvtColor(BGR_image, cv.COLOR_BGR2HSV)
    # Define a green mask (you might need to adjust the HSV ranges)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    
    # Find contours on the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) < 1000:
            continue  # Skip small contours
        # Approximate the contour to simplify its shape
        epsilon = 0.03 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        
        # This is a very simplified check: if the approximated contour has 7 points,
        # it might be an arrow. You might instead use more advanced shape matching.
        if len(approx) == 7:
            # Decide arrow direction based on the centroid position relative to image center
            M = cv.moments(cnt)
            if M['m00'] == 0:
                continue
            cX = int(M['m10'] / M['m00'])
            width = BGR_image.shape[1]
            if cX < width // 2:
                return "turn_left"
            else:
                return "turn_right"
    return None

def decide_command(BGR_image, counts):

    total_pixels = BGR_image.shape[0] * BGR_image.shape[1]
    
    # Set thresholds as a fraction of the image area
    red_threshold = 0.15 * total_pixels
    green_threshold = 0.15 * total_pixels
    yellow_threshold = 0.15 * total_pixels

    try:
        red_index = np.where(color_names == "red")[0][0]
        green_index = np.where(color_names == "green")[0][0]
        yellow_index = np.where(color_names == "yellow")[0][0]
    except IndexError:
        # In case the color is not in the list, default to no command
        return None

    if counts[red_index] > red_threshold:
        return "stop"
    elif counts[yellow_index] > yellow_threshold:
        return "slow"
    elif counts[green_index] > green_threshold:
        # Check if a green arrow is present
        arrow_cmd = detect_arrow_direction(BGR_image)
        if arrow_cmd:
            return arrow_cmd
        else:
            return "start"
    else:
        return None

class image_processor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.get_logger().info("Image processor node started!")
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  
            self.image_callback,
            10)
        self.command_pub = self.create_publisher(String, 'control_signal', 10)
        self.bridge = CvBridge()
        self.CUDA = torch.cuda.is_available()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error("CvBridge error: " + str(e))
            return

        if self.CUDA:
            processed_img, counts = descritize_image_GPU(frame)
        else:
            processed_img, counts = descritize_image_CPU(frame)
        
        command = decide_command(frame, counts)
        if command:
            out_msg = String()
            out_msg.data = command
            self.command_pub.publish(out_msg)
            self.get_logger().info(f"Published command: {command}")

        # Optionally show the processed image for debugging
        cv.imshow("Processed Feed", processed_img)
        cv.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = image_processor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

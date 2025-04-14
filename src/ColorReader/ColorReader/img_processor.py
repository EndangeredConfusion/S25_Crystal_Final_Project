import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import cv2 as cv
import numpy as np
import os
from sklearn.neighbors import KDTree
import torch
import webcolors


# Define the basic web color names (you can adjust the selection as needed)
web_color_names = list(webcolors.HTML4_NAMES_TO_HEX.keys())

# Convert web color names to RGB
web_colors_rgb = np.array([webcolors.name_to_rgb(color_name) for color_name in web_color_names])

# Prepare your color labels as enum or simple list (to use as identifiers)
color_names = np.array(web_color_names)
label_indices = np.array(range(len(web_color_names)))

# Convert web RGB to BGR (for OpenCV compatibility)
reference_BGR_colors = {color_name: webcolors.name_to_rgb(color_name) for color_name in web_color_names}
reference_BGR_colors = {k: (v[2], v[1], v[0]) for k, v in reference_BGR_colors.items()}  # Convert RGB to BGR

# Precompute the CIELAB values for the selected web colors
precomputed_CIELAB_values = np.array([
    cv.cvtColor(
        np.uint8([[reference_BGR_colors[label]]]),  # BGR values
        cv.COLOR_BGR2LAB
    ).astype(np.float32)[0, 0]
    for label in color_names
])

# Precompute KDTree for fast nearest neighbor search
color_tree = KDTree(precomputed_CIELAB_values)

# Use the same color labels but derived from CSS4 color names
color_names = np.array(color_names)

# Predefined color values for comparison (in BGR format)
color_values = np.array([reference_BGR_colors[label] for label in color_names])

def descritize_image_CPU(BGR_image):
    CEILAB_pixels = cv.cvtColor(BGR_image, cv.COLOR_BGR2LAB).astype(np.float32)
    h, w, *_ = CEILAB_pixels.shape

    # Reshape the image for vectorized processing
    CEILAB_image = CEILAB_pixels.reshape(-1, 3)

    # Nearest neighbor search with KDTree
    _, nearest_indices = color_tree.query(CEILAB_image, k=1)

    # Map the indices back to the corresponding color values (BGR)
    img_with_values = color_values[nearest_indices.flatten()]

    # Reshape the result to the original image shape
    output_image = img_with_values.reshape(h, w, 3).astype(np.uint8)

    # Count the occurrences of each color
    # counts = np.bincount(nearest_indices, minlength=len(color_names))

    # return output_image, counts
    return output_image, []


def descritize_image_GPU(BGR_image):
    img = cv.convertScaleAbs(BGR_image, alpha=1.5, beta=0)
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    h, w, _ = img_lab.shape

    img_lab_flat = torch.tensor(img_lab.reshape(-1, 3), device='cuda')

    lab_refs = torch.tensor(precomputed_CIELAB_values, device='cuda')

    diffs = img_lab_flat[:, None, :] - lab_refs[None, :, :]
    distances = torch.norm(diffs, dim=2)

    nearest_indices = torch.argmin(distances, dim=1)

    rgb_refs = torch.tensor(color_values, device='cuda')

    recolored_flat = rgb_refs[nearest_indices]

    # output 1
    recolored = recolored_flat.view(h, w, 3).cpu().numpy().astype(np.uint8)
    # output 2
    counts = torch.bincount(nearest_indices, minlength=len(color_names)).cpu()

    return recolored, counts



class ImgProcessor(Node):
    def __init__(self):
        super().__init__('img_processor')
        ### Uncomment if not using camera, comment out otherwise
        # self.timer = self.create_timer(1.0, self.process_image)  # Run every second change rate in the future for real time
        ###
        self.get_logger().info("Image processor node started!")
        self.CUDA = torch.cuda.is_available()
        self.counts_pub = self.create_publisher(Int32MultiArray, 'color_counts', 10)
        ### Comment out if not using camera, uncomment otherwise
        self.process_camera_feed()
        ###

    def process_camera_feed(self):
        # Open the default camera
        cap = cv.VideoCapture(0)

        # Check if the camera opened successfully
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame is read correctly, ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Process frame
            if self.CUDA:
                img, counts = descritize_image_GPU(frame)
            else:
                img, counts = descritize_image_CPU(frame)
            
            # Display the resulting frame
            cv.imshow('Camera Feed', img)

            # Press 'q' to exit
            if cv.waitKey(1) == ord('q'):
                break

        # When everything is done, release the capture
        cap.release()
        cv.destroyAllWindows()
    
    def process_image(self):
        
        img_path = '/camera_shared/frame.jpg'
        if not os.path.exists(img_path):
            self.get_logger().warn("Image file not found")
            return

        img = cv.imread(img_path)
        if img is None:
            self.get_logger().warn("Failed to read image")
            return

        # Example processing: convert to grayscale
        if self.CUDA:
            img, counts = descritize_image_GPU(img)
        else:
            img, counts = descritize_image_CPU(img)
        
        cv.imwrite('/camera_shared/discretized.jpg', img)
        self.get_logger().info("Saved processed image to /camera_shared/discretized.jpg")

        msg = Int32MultiArray()
        msg.data = counts.tolist()
        self.counts_pub.publish(msg)
        self.get_logger().info("Published color count array.")


def main(args=None):
    rclpy.init(args=args)
    node = ImgProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

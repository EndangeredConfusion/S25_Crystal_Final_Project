import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import cv2 as cv
import numpy as np
import os
from sklearn.neighbors import KDTree
import torch
import webcolors


<<<<<<< HEAD
class Colors(enum.Enum):
    BLACK = 0
    WHITE = 1
    GREY = 2
    RED = 3
    ORANGE = 4
    YELLOW = 5
    GREEN = 6
    BLUE = 7
    PURPLE = 8
color_labels = np.array(list(Colors))
label_indices = np.array(range(len(Colors)))


reference_RGB_colors = {
    Colors.BLACK: (0, 0, 0),
    Colors.WHITE: (255, 255, 255),
    Colors.GREY: (128, 128, 128),
    Colors.RED: (255, 0, 0),
    Colors.ORANGE: (255, 165, 0),
    Colors.YELLOW: (255, 255, 0),
    Colors.GREEN: (0, 255, 0),
    Colors.BLUE: (0, 0, 255),
    Colors.PURPLE: (128, 0, 128)}

reference_BGR_colors = {
    Colors.BLACK: (0, 0, 0),
	Colors.WHITE: (255, 255, 255),
	Colors.GREY: (128, 128, 128),
	Colors.RED: (0, 0, 255),
	Colors.ORANGE: (0, 165, 255),
	Colors.YELLOW: (0, 255, 255),
	Colors.GREEN: (0, 255, 0),
	Colors.BLUE: (255, 0, 0),
	Colors.PURPLE: (128, 0, 128)
}
color_values = np.array([reference_BGR_colors[label] for label in color_labels])

''' Code to precompute values (low precision for now)
ref_lab = {
    label: cv.cvtColor(
        np.uint8([[rgb]]),
        cv.COLOR_BGR2LAB
    ).astype(np.float32)[0, 0]
    for label, rgb in reference_BGR_colors.items()
}
for label,color_val in ref_lab.items():
    print(f"\t{label}: {str(color_val).replace('. ', ', ').replace('.]', '],')}")
'''

=======
# Define the basic web color names (you can adjust the selection as needed)
web_color_names = webcolors.names()

# Convert web color names to RGB
web_colors_rgb = np.array([webcolors.name_to_rgb(color_name) for color_name in web_color_names])

# Prepare your color labels as enum or simple list (to use as identifiers)
color_names = np.array(web_color_names)
label_indices = np.array(range(len(web_color_names)))

# Convert web RGB to BGR (for OpenCV compatibility)
reference_BGR_colors = {color_name: webcolors.name_to_rgb(color_name) for color_name in web_color_names}
reference_BGR_colors = {k: (v[2], v[1], v[0]) for k, v in reference_BGR_colors.items()}  # Convert RGB to BGR

# Precompute the CIELAB values for the selected web colors
>>>>>>> 1e6d0a3539f32416ddba391901f0d613c5c090bf
precomputed_CIELAB_values = np.array([
    cv.cvtColor(
        np.uint8([[reference_BGR_colors[label]]]),  # BGR values
        cv.COLOR_BGR2LAB
    ).astype(np.float32)[0, 0]
    for label in color_names
])
<<<<<<< HEAD
=======

# Precompute KDTree for fast nearest neighbor search
color_tree = KDTree(precomputed_CIELAB_values)

# Use the same color labels but derived from CSS4 color names
color_names = np.array(color_names)

# Predefined color values for comparison (in BGR format)
color_values = np.array([reference_BGR_colors[label] for label in color_names])
>>>>>>> 1e6d0a3539f32416ddba391901f0d613c5c090bf

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

        self.get_logger().info("Image processor node started!")
        
        self.counts_pub = self.create_publisher(Int32MultiArray, 'color_counts', 10)

        self.CUDA = torch.cuda.is_available()

        self.declare_parameter("frame_rate", 60)

        frame_rate = self.get_parameter('frame_rate').get_parameter_value().double_value

        self.frame_period = 1/frame_rate
        self.get_logger().info(f"Image processor started with frame rate: {frame_rate} FPS")

        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
            exit()
        
        self.timer = self.create_timer(self.frame_period, self.process_frame)
        ### Comment out if not using camera, uncomment otherwise
        self.process_camera_feed()

    def process_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().error("Image not retrieved!")
            return
        
        if self.CUDA:
            img, counts = descritize_image_GPU(frame)
        else:
            img, counts = descritize_image_CPU(frame)         

        cv.imshow("Processed Feed", img)

        msg = Int32MultiArray()
        msg.data = counts.tolist()
        self.counts_pub.publish(msg)
        self.get_logger().info("Published color count array.")

        

    def process_camera_feed(self):
        # Open the default camera
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FPS, 10)

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

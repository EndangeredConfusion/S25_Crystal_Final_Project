import rclpy
from rclpy.node import Node
import cv2 as cv
import os
import enum
import numpy as np
import torch
from std_msgs.msg import Int32MultiArray


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
    PINK = 9
    CYAN = 10
    BROWN = 11
    LIGHT_BLUE = 12
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
    Colors.PURPLE: (128, 0, 128),
    Colors.PINK: (255, 105, 180),     
    Colors.CYAN: (0, 255, 255),       
    Colors.BROWN: (139, 69, 19),      
    Colors.LIGHT_BLUE: (173, 216, 230)
}

reference_BGR_colors = {
    Colors.BLACK: (0, 0, 0),
    Colors.WHITE: (255, 255, 255),
    Colors.GREY: (128, 128, 128),
    Colors.RED: (255, 0, 0),
    Colors.ORANGE: (255, 165, 0),
    Colors.YELLOW: (255, 255, 0),
    Colors.GREEN: (0, 255, 0),
    Colors.BLUE: (0, 0, 255),
    Colors.PURPLE: (128, 0, 128),
    Colors.PINK: (180, 105, 255),     
    Colors.CYAN: (255, 255, 0),       
    Colors.BROWN: (19, 69, 139),      
    Colors.LIGHT_BLUE: (230, 216, 173)
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
# precomputed_CIELAB_values_dict = {
# 	Colors.BLACK: [  0, 128, 128],
# 	Colors.WHITE: [255, 128, 128],
# 	Colors.GREY: [137, 128, 128],
# 	Colors.RED: [136, 208, 195],
# 	Colors.ORANGE: [191, 152, 207],
# 	Colors.YELLOW: [248, 106, 223],
# 	Colors.GREEN: [224,  42, 211],
# 	Colors.BLUE: [ 82, 207,  20],
# 	Colors.PURPLE: [ 76, 187,  91],
# }
precomputed_CIELAB_values = np.array([
    cv.cvtColor(
        np.uint8([[reference_BGR_colors[label]]]),  # this is BGR now
        cv.COLOR_BGR2LAB
    ).astype(np.float32)[0, 0]
    for label in color_labels
])
# precomputed_CIELAB_values = np.array([precomputed_CIELAB_values_dict[color] for color in color_labels])

def descritize_image_CPU(BGR_image):
    img = cv.convertScaleAbs(BGR_image, alpha=1.5, beta=0)
    CEILAB_pixels = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    h, w, *_ = CEILAB_pixels.shape

    # this looks like a 3d array with color values a, b, c
    '''
    [[[a,b,c], [a,b,c], [a,b,c]],
    [[a,b,c], [a,b,c], [a,b,c]]]
    '''
    # need it to look like this for vectorization
    '''
    [[abc], [abc], [abc], [abc], ...]
    '''
    # could do [px for row in CEILAB_image for px in row], but numpy is faster
    CEILAB_image = CEILAB_pixels.reshape(-1, 3)
    element_wise_diffs = CEILAB_image[ : ,np.newaxis, : ] - precomputed_CIELAB_values[ np.newaxis, : , : ]
    # given refrence color r_x
    '''
    [[[d1r1a, d1r1b, d1r1c], [d1r2a, d1r2b, d1r2c], [d1r3a, d1r3b, d1r3c], ...],
    [[d2r1a, d2r1b, d2r1c], [d2r2a, d2r2b, d2r2c], [d2r3a, d2r3b, d2r3c], ...]]
    '''
    distances = np.linalg.norm(element_wise_diffs, axis=2)  #innermost axis
    '''
    [[d1r1, d1r2, d1r3, ...],
    [d2r1, d2r2, d2r3, ...]]
    '''
    nearest_indicies = np.argmin(distances, axis=1)  # get index of minimum distance (corresponds to colors vector)
    '''
    [i1, i2, i3, i4, i5, i6, ...]
    '''
    img_with_values = color_values[nearest_indicies]
    '''
    [[r,g,b], [r,g,b], [r,g,b], ...]
    '''
    #output 1
    output_image = img_with_values.reshape(h, w, 3).astype(np.uint8)

    #output 2
    counts = np.bincount(nearest_indicies, minlength=len(Colors))

    return output_image, counts


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
    counts = torch.bincount(nearest_indices, minlength=len(Colors)).cpu()

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

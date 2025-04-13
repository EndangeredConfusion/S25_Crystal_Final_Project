# S25_Crystal_Final_Project

---
## Important Note
- Because I'm on Macos and docker and camera don't play nice the current node reads images from a mounted directory
- `~/camera_shared/frame.jpg` if you want to test on an image before camera functionality is added you can add the image to that folder, and you will get an output `/camera_shared/discretized.jpg`.

## TODO
- Custom message for color/counts (smth like this)
```
ColorHistogram.msg
string[] color_labels
int32[] color_counts
```
- Interface with light bar to display the colors
- Sample at higher rate for real time processing
- Add color list voice command
- ...


## Setup & Build Instructions

### Requirements

- ROS 2 (Foxy)
- `colcon` build tool
- Python 3.8+ (included with ROS 2)
- ROS 2 image and std_msgs dependencies

---

### Project Structure

```
ros2_final_proj/
├── src/
│   └── ColorReader/
│       ├── ColorReader/        # Main Python module
│       ├── resource/
│       ├── test/               # Unit & lint tests
│       ├── setup.py
│       ├── setup.cfg
│       └── package.xml
├── build/                      # Ignored build artifacts
├── install/                    # Ignored install artifacts
├── log/                        # Ignored logs
├── .gitignore
└── README.md
```

---

### Build the Workspace

```bash
# Clone this repository
git clone https://github.com/EndangeredConfusion/S25_Crystal_Final_Project.git
cd S25_Crystal_Final_Project

# Source your ROS 2 environment
source /opt/ros/foxy/setup.bash

# Build the workspace
colcon build

# Source again
source install/setup.bash

# After you build and source the project you should be able to do install the Python dependencies (torch might take a while)
cd src/ColorReader
pip3 install .
```

---

### Run the Node

```bash
# Source the overlay workspace
source install/setup.bash

# Run the node (update the name if you have a specific entry point)
ros2 run color_reader img_processor_node

# In a seperate terminal
ros2 topic echo /color_counts
```

---

## Developer Notes

### GitHub Collaboration

- Ensure `.gitignore` is respected to avoid committing `build/`, `install/`, `log/`, and Python cache files.
- Created using VS Code with dev containers, `.devcontainer/` is supported but ignored by Git.


### Contributors

- Kaeshev Alapati (@EndangeredConfusion)
- Marilla Bongiovanni (@bongim5)
- Eric Carson (@carsoe2)
- Add your names

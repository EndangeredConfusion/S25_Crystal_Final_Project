<h2> Gazebo Simulation</h2>
The .urdf.xacro file defines a robot with 4 wheels, and a differential_drive_controller that allows you to move it like teleop_turtle from the labs.
The easiest way to run this is to duplicate your Lab 7, and replace your .urdf.xacro file with this new one, and run the launch file like we did in the lab.
The reason the original robot from Lab 7 isn't chosen is because I spent hours of it to realize the physics of the wheels doesn't let the robot turn at all, so I just made a new one.
That should open:

<p align="left">
  <img  src="https://github.com/user-attachments/assets/033bf571-d59a-4588-b430-81891617047f" style="width: 50%; height: auto;">
</p>

Once it's open, run `ros2 run teleop_twist_keyboard teleop_twist_keyboard` to open up:

<p align="left">
  <img  src="https://github.com/user-attachments/assets/f693d0ef-eaed-474e-b878-581e5ec82a91" style="width: 15%; height: auto;">
</p>

Play around with it

<h2> Speed Changing Node </h2>
This works by subscribing to /color_counts and publishing to /cmd_vel to control the speed of the robot.
I tried to make it as real as possible so this code could realistically be run on a ROSMASTER X3 without any changes, even though for the purposes of this project it's a simualtion because I don't wanna deal with the robots.
But for now, I just type in 

<code>
GREEN 63
ros2 topic pub /color_counts std_msgs/msg/Int32MultiArray "$(python3 -c 'counts=[0]*140; counts[63]=5; print(f"{{data: {counts}}}")')" --once
RED 118
ros2 topic pub /color_counts std_msgs/msg/Int32MultiArray "$(python3 -c 'counts=[0]*140; counts[118]=5; print(f"{{data: {counts}}}")')" --once
YELLOW 139
ros2 topic pub /color_counts std_msgs/msg/Int32MultiArray "$(python3 -c 'counts=[0]*140; counts[139]=5; print(f"{{data: {counts}}}")')" --once
</code>

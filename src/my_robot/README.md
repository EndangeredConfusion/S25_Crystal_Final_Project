Run instructions:
open a terminal under the S25_CRYSTAL_FINAL_PROJECT directory, run `colcon build` and `source ./install/setup.bash`, then run `ros2 launch my_robot my_robot_gazebo_rviz.launch.py`

What should happen: 
- the robot model should be built in gazebo and start moving in a circle
- every 5 seconds, a "light" (color box) will appear above the robot as red or green
- depending on the color, the robot will stop or keep moving in a circle

Important files:
- urdf/my_robot.urdf.xacro
    - Expanded on Rica's file to fix steering
- models (all files)
    - Created red/green boxes which are created in gazebo after launch to represent red/green lights
- setup.py
    - Added sky_light_switcher to list of entry points
- launch/my_robot_gazebo_rviz_launch.py
    - Removed rviz launch and added light switcher node
- my_robot/sky_light_switcher.py
    - spawns the red/green boxes and activates/deactivates robot motion depending on robot color
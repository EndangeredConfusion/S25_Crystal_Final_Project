Run instructions:
open two terminals under the S25_CRYSTAL_FINAL_PROJECT directory, in both run `colcon build` and `source ./install/setup.bash`
- in one terminal run `ros2 launch my_robot my_robot_gazebo_rviz.launch.py`
- in the other run `ros2 run teleop_twist_keyboard teleop_twist_keyboard`
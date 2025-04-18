import rospy
from gazebo_msgs.srv import SetLightProperties
from std_msgs.msg import ColorRGBA

def set_light(color):
    rospy.wait_for_service('/gazebo/set_light_properties')
    try:
        set_light = rospy.ServiceProxy('/gazebo/set_light_properties', SetLightProperties)
        set_light('sun', ColorRGBA(color[0], color[1], color[2], 1.0), 1.0, 1000.0, 1000.0)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

def switch_lights():
    rate = rospy.Rate(0.5)  # every 2 seconds
    green = (0.0, 1.0, 0.0)
    red = (1.0, 0.0, 0.0)
    toggle = True
    while not rospy.is_shutdown():
        color = green if toggle else red
        set_light(color)
        toggle = not toggle
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('red_green_light_controller')
    switch_lights()

cmake_minimum_required(VERSION 2.8.3)
project(ss_segmentation)

find_package(catkin REQUIRED)

catkin_package(CATKIN_DEPENDS std_msgs sensor_msgs)

catkin_python_setup()

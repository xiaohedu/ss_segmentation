cmake_minimum_required(VERSION 2.8.3)
project(ss_segmentation)

find_package(catkin REQUIRED)

catkin_package(CATKIN_DEPENDS sensor_msgs)

catkin_python_setup()

catkin_install_python(PROGRAMS nodes/ss_inference
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

cmake_minimum_required(VERSION 3.0.2)
project(kamayuc_perception)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  aruco
  aruco_msgs
  cv_bridge
  image_transport
  message_generation
)

find_package(OpenCV)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES kamayuc_navigation
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(read_video_cpp src/read_video.cpp)
target_link_libraries(read_video_cpp ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


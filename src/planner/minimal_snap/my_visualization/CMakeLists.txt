cmake_minimum_required(VERSION 3.0.2)
project(my_visualization)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  quadrotor_msgs
  traj_utils
)

find_package(Eigen3 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES my_visualization
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS} 
)

# add_library( plan_visual
#   include/my_visualization/plan_visual.h
#   src/plan_visual.cpp
# )
# target_link_libraries(plan_visual ${catkin_LIBRARIES})

# add_executable(obvp_visual src/obvp_visual.cpp)
# target_link_libraries(obvp_visual plan_visual)

add_executable(min_snap_visual src/min_snap_visual.cpp src/plan_visual.cpp)
target_link_libraries(min_snap_visual ${catkin_LIBRARIES})

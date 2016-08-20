cmake_minimum_required(VERSION 2.8.3)
project(visual_odom)

find_package(catkin REQUIRED)
find_package(catkin REQUIRED COMPONENTS std_msgs tf sensor_msgs image_transport cv_bridge camera_info_manager pcl_ros pcl_conversions)
find_package( OpenCV REQUIRED )
find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(visual_odom src/visual_odom.cpp) 
target_link_libraries(visual_odom ${catkin_LIBRARIES}  ${OpenCV_LIBS})

cmake_minimum_required(VERSION 2.8.3)
project(visual_odom)

find_package(catkin REQUIRED)
find_package(catkin REQUIRED COMPONENTS std_msgs tf sensor_msgs image_transport cv_bridge camera_info_manager pcl_ros pcl_conversions)
find_package( OpenCV REQUIRED )
find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)
find_package(Ceres REQUIRED)

catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

add_executable(visual_odom
    src/visual_odom.cpp src/keyframe.cpp src/camera_model.cpp) 
target_link_libraries(visual_odom ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES})

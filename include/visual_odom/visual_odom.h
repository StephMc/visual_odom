#ifndef VISUAL_ODOM_H
#define VISUAL_ODOM_H

#include <visual_odom/camera_model.h>
#include <visual_odom/keyframe.h>
#include <sensor_msgs/Imu.h>
#include <boost/thread/locks.hpp>

class VisualOdom
{
public:
  VisualOdom(ros::NodeHandle &nh);

private:
  void publishPointCloud(std::vector<Eigen::Vector4d> &points,
    std::string frame, ros::Publisher &pub);

  void drawPoints(cv::Mat& frame,
    std::vector<cv::Point2f> points, cv::Scalar color);

  Eigen::Matrix3d create_rotation_matrix(double ax, double ay, double az);

  void publishTransform(Eigen::Matrix4d& pose,
    std::string parent, std::string child);

  void callback(const sensor_msgs::ImageConstPtr& left_image,
      const sensor_msgs::ImageConstPtr& right_image);

  void imuCallback(const sensor_msgs::Imu::ConstPtr& imu);

  bool need_new_keyframe_;
  Eigen::Matrix4d keyframe_pose_;
  Eigen::Matrix4d camera_pose_;

  // Publishers
  ros::Publisher cloud_pub_, debug_cloud_pub_, keyframe_cloud_pub_,
      odom_pub_;
  tf::TransformBroadcaster br_;

  // Subscribers
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image> ImageSyncPolicy;
  message_filters::Subscriber<sensor_msgs::Image> left_sub_, right_sub_;
  message_filters::Synchronizer<ImageSyncPolicy> sync_;
  ros::Subscriber imu_sub_single_;
  
  int max_feature_count_;

  CameraModel camera_model_;

  Keyframe *curr_keyframe_, *prev_keyframe_;

  sensor_msgs::Imu imu_data_;
  bool imu_init_;
  boost::mutex imu_mutex_;
};

#endif

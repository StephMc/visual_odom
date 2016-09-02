#ifndef VISUAL_ODOM_H
#define VISUAL_ODOM_H

#include <visual_odom/camera_model.h>
#include <visual_odom/keyframe.h>

class VisualOdom
{
public:
  VisualOdom(ros::NodeHandle &nh);

private:
  void publishPointCloud(std::vector<Eigen::Vector4d> &points,
    std::string frame, ros::Publisher &pub);

  void callback(const sensor_msgs::ImageConstPtr& left_image,
      const sensor_msgs::ImageConstPtr& right_image);

  bool need_new_keyframe_;

  // Publishers
  ros::Publisher cloud_pub_, debug_cloud_pub_, keyframe_cloud_pub_;
  tf::TransformBroadcaster br_;

  // Subscribers
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image> ImageSyncPolicy;
  message_filters::Subscriber<sensor_msgs::Image> left_sub_, right_sub_;
  message_filters::Synchronizer<ImageSyncPolicy> sync_;
  
  int max_feature_count_;

  CameraModel camera_model_;

  Keyframe *curr_keyframe_, *prev_keyframe_;
};

#endif

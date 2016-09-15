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

  void drawPoints(cv::Mat& frame,
    std::vector<cv::Point2f> points, cv::Scalar color);

  Eigen::Matrix3d create_rotation_matrix(double ax, double ay, double az);

  void removeFeatures(std::vector<cv::Point2f> &lpoints,
    std::vector<cv::Point2f> &rpoints, std::vector<cv::Point2f> &cpoints,
    std::vector<uchar> &status);

  Eigen::Matrix4d getPoseDiffKeyframe(
    std::vector<cv::Point2f>& prevKeyPoints,
    std::vector<Eigen::Vector4d>& curr3dPoints,
    std::vector<cv::Point2f>& currKeyPoints,
    std::vector<Eigen::Vector4d>& prev3dPoints);

  std::vector<cv::Point2f> calculate3dPoints(cv::Mat& keyframe,
    std::vector<cv::Point2f> keyframe_features, cv::Mat &lframe,
    cv::Mat &rframe, std::vector<Eigen::Vector4d> &points3d);

  void callback(const sensor_msgs::ImageConstPtr& left_image,
      const sensor_msgs::ImageConstPtr& right_image);

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
  
  int max_feature_count_;

  CameraModel camera_model_;

  Keyframe *curr_keyframe_, *prev_keyframe_;
};

#endif

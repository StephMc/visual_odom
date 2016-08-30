#ifndef VISUAL_ODOM_H
#define VISUAL_ODOM_H

class VisualOdom
{
public:
  VisualOdom(ros::NodeHandle &nh);

  struct camera_params
  {
    double k1, k2, cx, cy, f, b, rx, ry, rz;
  };
  typedef struct camera_params CameraParams;

  struct camera_pose
  {
    double x, y, z, rx, ry, rz;
  };
  typedef struct camera_pose CameraPose;

private:
  Eigen::Matrix3d create_rotation_matrix(double ax, double ay, double az);

  void findPlotPoints(cv::Mat &grey, cv::Mat &prevGrey,
      std::vector<cv::Point2f> points, cv::Mat &frame, cv::Scalar color,
      std::vector<cv::Point2f> &output, std::vector<bool> &errors);

  void calculate3dPoints(std::vector<Eigen::Vector4d> &points3d,
      std::vector<cv::Point2f> points2d[], cv::Point2f midpoint);

  void publishPointCloud(std::vector<Eigen::Vector4d> &points,
    std::string frame, ros::Publisher &pub);

  void correctRadial(std::vector<cv::Point2f> &points, double k1,
      double k2, double cx, double cy, double f,
      cv::Scalar color, cv::Mat &frame);

  void correctRotation(std::vector<cv::Point2f> &points,
      cv::Scalar color, cv::Mat &frame);

  void detectBadMatches(std::vector<cv::Point2f> &lp,
      std::vector<cv::Point2f> &rp, std::vector<bool> &errors);

  Eigen::Matrix4d getPoseDiff(std::vector<Eigen::Vector4d> &currPoints,
      std::vector<Eigen::Vector4d> &keyframePoints);
 
  Eigen::Matrix4d getPoseDiffImageSpace(
    std::vector<cv::Point2f> prevPoints,
    std::vector<Eigen::Vector4d> currPoints);

  void callback(const sensor_msgs::ImageConstPtr& left_image,
      const sensor_msgs::ImageConstPtr& right_image);

  std::vector<cv::Point2f> opoints, lpoints, lrpoints[2];
  std::vector<Eigen::Vector4d> prevPoints3d;
  cv::Mat prevLGrey, prevRGrey;
  bool needToInit;
  //Eigen::Matrix4d pose, basePose;

  // Publishers
  ros::Publisher cloudPub, debugCloudPub, keyframeCloudPub;

  // Subscribers
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image> ImageSyncPolicy;
  message_filters::Subscriber<sensor_msgs::Image> left_sub, right_sub;
  message_filters::Synchronizer<ImageSyncPolicy> sync;
  
  // LK flow params
  cv::TermCriteria termcrit;
  cv::Size subPixWinSize, winSize;
  
  // Max number of feature to track
  int max_count;

  CameraParams l_cam_params, r_cam_params;
  CameraPose currPose, basePose;
};

#endif

#ifndef VISUAL_ODOM_H
#define VISUAL_ODOM_H

class VisualOdom
{
public:
  VisualOdom();

private:
  Eigen::Matrix3d create_rotation_matrix(double ax, double ay, double az);

  void findPlotPoints(cv::Mat &grey, cv::Mat &prevGrey,
      std::vector<cv::Point2f> points[], cv::Mat &frame, cv::Scalar color,
      std::vector<cv::Point2f> &output);

  void calculate3dPoints(std::vector<Eigen::Vector3d> &points3d,
      std::vector<cv::Point2f> points2d[], cv::Point2f midpoint);

  void publishPointCloud(std::vector<Eigen::Vector3d> &points,
    std::string frame, ros::Publisher &pub);

  void correctRadial(std::vector<cv::Point2f> &points, double k1,
      double k2, double cx, double cy, double fx, double fy,
      cv::Scalar color, cv::Mat &frame);

  void correctRotation(std::vector<cv::Point2f> &points,
      cv::Scalar color, cv::Mat &frame);
  
  void callback(const sensor_msgs::ImageConstPtr& left_image,
      const sensor_msgs::ImageConstPtr& right_image,
      const geometry_msgs::QuaternionStampedConstPtr& imu_rotation);

  std::vector<cv::Point2f> lpoints[2], lrpoints[2];
  cv::Mat prevLGrey, prevRGrey;
  bool needToInit;

  // Publishers
  ros::Publisher cloudPub, debugCloudPub, keyframeCloudPub;
  
  // LK flow params
  cv::TermCriteria termcrit;
  cv::Size subPixWinSize, winSize;
  
  // Max number of feature to track
  int max_count;
};

#endif

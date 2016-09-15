#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <vector>
#include <visual_odom/camera_model.h>

class Keyframe
{
public:
  Keyframe(cv::Mat &lframe, cv::Mat &rframe, CameraModel& camera_model,
      int max_feature_count);

  Eigen::Matrix4d getRelativePose(cv::Mat& lframe, cv::Mat& rframe);

  Eigen::Matrix4d getGroundRelativePose();

  cv::Mat& getKeyframe();
  cv::Mat& getKeyframeRight();

  std::vector<cv::Point2f>& getKeyframeRawFeatures();
  std::vector<cv::Point2f>& getKeyframeCorrectedFeatures();

  std::vector<Eigen::Vector4d>& getRecent3d();
  std::vector<Eigen::Vector4d>& getKeyframe3d();
private:
  void calculate3dPoints(cv::Mat &lframe, cv::Mat &rframe,
    std::vector<Eigen::Vector4d> &points3d);

  Eigen::Matrix3d create_rotation_matrix(double ax, double ay, double az);

  Eigen::Matrix4d getPoseDiffImageSpace(
    std::vector<cv::Point2f>& prevPoints,
    std::vector<Eigen::Vector4d>& currPoints);

  void removeFeatures(std::vector<cv::Point2f> &lpoints,
    std::vector<cv::Point2f> &rpoints, std::vector<uchar> &status);

  void drawPoints(cv::Mat& frame,
      std::vector<cv::Point2f> points, cv::Scalar color);

  // LK flow params
  cv::TermCriteria termcrit;
  cv::Size subPixWinSize, winSize;
  
  // local copy of the keyframe
  cv::Mat keyframe_;
  cv::Mat keyframe_right_;

  // local copy of the last frame
  cv::Mat recent_;

  // Uncorrected feature on keyframe
  std::vector<cv::Point2f> raw_features_;

  // Corrected feature on keyframe
  std::vector<cv::Point2f> corrected_features_;

  // Most recent, uncorrected observation of feature
  std::vector<cv::Point2f> recent_features_;

  // 3D estimate of feature using the keyframe for debug
  std::vector<Eigen::Vector4d> keyframe3d_;

  // 3D estimate of feature using the keyframe for debug
  std::vector<Eigen::Vector4d> recent3d_;

  CameraModel& camera_model_;
};
#endif

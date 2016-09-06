#include <visual_odom/keyframe.h>
#include <visual_odom/camera_model.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

Keyframe::Keyframe(cv::Mat &lframe, cv::Mat &rframe,
    CameraModel& camera_model, int max_feature_count) :
    termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03),
    subPixWinSize(10,10),
    winSize(31,31),
    camera_model_(camera_model)
{
  cv::goodFeaturesToTrack(lframe, raw_features_, max_feature_count, 0.01,
      10, cv::Mat(), 3, 0, 0.04);
  cornerSubPix(lframe, raw_features_, subPixWinSize, cv::Size(-1,-1),
      termcrit);
  
  corrected_features_.resize(raw_features_.size());
  camera_model_.correctRadial(raw_features_, corrected_features_,
      CameraModel::LEFT_CAMERA);

  recent_features_ = raw_features_;
  lframe.copyTo(keyframe_);
  lframe.copyTo(recent_);
  rframe.copyTo(keyframe_right_);

  keyframe3d_.resize(raw_features_.size());
  recent3d_.resize(raw_features_.size());
  calculate3dPoints(lframe, rframe, keyframe3d_); 
}

void Keyframe::calculate3dPoints(cv::Mat &lframe, cv::Mat &rframe,
    std::vector<Eigen::Vector4d> &points3d)
{
  std::vector<uchar> status1, status2;
  std::vector<float> err;
  std::vector<cv::Point2f> lpoints, rpoints;
  lpoints.resize(recent_features_.size());
  rpoints.resize(recent_features_.size());

  // Match between last frame and current frame
  cv::calcOpticalFlowPyrLK(
      recent_, lframe, recent_features_, lpoints, status1, err, winSize,
      3, termcrit, 0, 0.001);
  removeFeatures(lpoints, rpoints, status1);

  // Match between current left and right frame
  cv::calcOpticalFlowPyrLK(
      lframe, rframe, lpoints, rpoints, status2, err, winSize, 3,
      termcrit, 0, 0.001);
  removeFeatures(lpoints, rpoints, status2);

  // Save the left frame and found features to use next cycle
  lframe.copyTo(recent_);
  recent_features_ = lpoints;

  // Correct points for distortion
  camera_model_.correctRadial(lpoints, lpoints, CameraModel::LEFT_CAMERA);
  camera_model_.correctRadial(rpoints, rpoints,
      CameraModel::RIGHT_CAMERA);
  camera_model_.correctRotation(rpoints, rpoints);

  // Remove matches not on epipolar line
  std::vector<uchar> status;
  for (int i = 0; i < lpoints.size(); ++i)
  {
    status.push_back(fabs(lpoints[i].y - rpoints[i].y) < 5 ? 1 : 0);
  }
  removeFeatures(lpoints, rpoints, status);

  // Debug display
  cv::Mat ld, rd;
  lframe.copyTo(ld);
  rframe.copyTo(rd);
  drawPoints(ld, recent_features_, cv::Scalar(255, 255, 255));
  drawPoints(ld, lpoints, cv::Scalar(0, 255, 0));
  cv::imshow("left", ld);
  for (int i = 0; i < lpoints.size(); ++i)
  {
    cv::line(rd, lpoints[i], rpoints[i], cv::Scalar(255, 255, 255),
        2, 8);
  }
  cv::imshow("right", rd);
  cv::waitKey(1);

  camera_model_.calculate3dPoints(points3d, lpoints, rpoints); 
}

Eigen::Matrix4d Keyframe::getRelativePose(cv::Mat &lframe,
    cv::Mat& rframe)
{
  calculate3dPoints(lframe, rframe, recent3d_);
  return getPoseDiffImageSpace(corrected_features_, recent3d_);
}

void Keyframe::removeFeatures(std::vector<cv::Point2f> &lpoints,
    std::vector<cv::Point2f> &rpoints, std::vector<uchar> &status)
{
  for (int i = lpoints.size() - 1; i >= 0; --i)
  {
    if (status[i]) continue;
    lpoints.erase(lpoints.begin() + i);
    rpoints.erase(rpoints.begin() + i);
    recent_features_.erase(recent_features_.begin() + i);
    corrected_features_.erase(corrected_features_.begin() + i);
    raw_features_.erase(raw_features_.begin() + i);
    recent3d_.erase(recent3d_.begin() + i);
    keyframe3d_.erase(keyframe3d_.begin() + i);
  }
}

void Keyframe::drawPoints(cv::Mat& frame, std::vector<cv::Point2f> points,
    cv::Scalar color)
{
  for (int i = 0; i < points.size(); ++i)
  {
    cv::circle(frame, points[i], 3, color, -1, 8);
  }
}

struct ImageDistResidual {
  ImageDistResidual(cv::Point2f orig, Eigen::Vector4d p,
      CameraModel::CameraParams cparams)
      : orig_(orig), p_(p), cparams_(cparams) {}
  template <typename T> bool operator()(const T* const rotation,
                                        const T* const translation,
                                        T* residual) const
  {
    // Rotate a by the passed in rotation and translation
    T R[9];
    /*T rot[3];
    rot[0] = rotation[0] / T(2);
    rot[1] = rotation[1] / T(2);
    rot[2] = rotation[2] / T(2);*/
    ceres::EulerAnglesToRotationMatrix<T>(rotation, 3, R);
    T x = R[0] * T(p_(0)) + R[1] * T(p_(1)) + R[2]* T(p_(2))
        - translation[0];
    T y = R[3] * T(p_(0)) + R[4] * T(p_(1)) + R[5]* T(p_(2))
        - translation[1];
    T z = R[6] * T(p_(0)) + R[7] * T(p_(1)) + R[8]* T(p_(2))
        - translation[2];
   
    // Convert back into 2d space
    T xl = ((-y * T(cparams_.f)) / x) + T(cparams_.cx);
    T yl = ((-z * T(cparams_.f)) / x) + T(cparams_.cy);

    // Calculate the error between reprojected points
    T ex = xl - T(orig_.x);
    T ey = yl - T(orig_.y);
    residual[0] = ex * ex;
    residual[1] = ey * ey;
    return true;
  }
 private:
  const cv::Point2f orig_;
  const Eigen::Vector4d p_;
  const CameraModel::CameraParams cparams_;
};

Eigen::Matrix4d Keyframe::getPoseDiffImageSpace(
    std::vector<cv::Point2f>& prevPoints,
    std::vector<Eigen::Vector4d>& currPoints)
{
  ceres::Problem problem;
  CameraModel::CameraParams cp = camera_model_.getAverageCamera();
  // TODO: Test if providing a better estimate helps
  // e.g. last cycle's estimate
  double rotation[3] = {0, 0, 0};
  double translation[3] = {0, 0, 0};
  for (int i = 0; i < currPoints.size(); ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ImageDistResidual, 2, 3, 3>(
          new ImageDistResidual(prevPoints[i], currPoints[i], cp)),
        new ceres::HuberLoss(1.0), rotation, translation);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //ROS_WARN_STREAM("Summary: " << summary.BriefReport());
  ROS_ERROR_STREAM("Rotation" << rotation[0] << " " <<
      rotation[1] << " " << rotation[2]);
  ROS_ERROR_STREAM("Translation" << translation[0] << " "
      << translation[1] << " " << translation[2]);
  Eigen::Affine3d r(create_rotation_matrix(rotation[0] * (M_PI / 180.0),
        rotation[1] * (M_PI / 180.0), rotation[2] * (M_PI / 180.0)));
  Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(-translation[0],
          -translation[1], -translation[2])));
  Eigen::Matrix4d pose = (t * r).matrix();
  return pose;
}

Eigen::Matrix3d Keyframe::create_rotation_matrix(double ax, double ay, double az) {
  Eigen::Matrix3d rx =
      Eigen::Matrix3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Matrix3d ry =
      Eigen::Matrix3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Matrix3d rz =
      Eigen::Matrix3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}

std::vector<Eigen::Vector4d>& Keyframe::getRecent3d()
{
  return recent3d_;
}

std::vector<Eigen::Vector4d>& Keyframe::getKeyframe3d()
{
  return keyframe3d_;
}

cv::Mat& Keyframe::getKeyframe()
{
  return keyframe_;
}

cv::Mat& Keyframe::getKeyframeRight()
{
  return keyframe_right_;
}

std::vector<cv::Point2f>& Keyframe::getKeyframeRawFeatures()
{
  return raw_features_;
}

std::vector<cv::Point2f>& Keyframe::getKeyframeCorrectedFeatures()
{
  return corrected_features_;
}


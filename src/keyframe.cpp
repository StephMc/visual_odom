#include <visual_odom/keyframe.h>
#include <visual_odom/camera_model.h>
#include <visual_odom/ceres_extensions.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <tf/transform_datatypes.h>

Keyframe::Keyframe(cv::Mat &lframe, cv::Mat &rframe,
    CameraModel& camera_model, int max_feature_count,
    sensor_msgs::Imu& imu, Eigen::Matrix4d& keyframe_pose) :
    termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03),
    subPixWinSize(10, 10),
    winSize(31, 31), is_lost_(false),
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
  keyframe_ok_ = calculate3dPoints(lframe, rframe, keyframe3d_); 
  
  // TODO: extract std_dev
  Eigen::Matrix3d rotMat = keyframe_pose.block<3, 3>(0, 0);
  keyframe_orientation_ = Eigen::Quaterniond(rotMat);
  //keyframe_orientation_ = Eigen::Quaterniond(
  //    imu.orientation.w, imu.orientation.x,
  //    imu.orientation.y, imu.orientation.z);

  ROS_ERROR_STREAM("Initial rot is " <<
      keyframe_orientation_.toRotationMatrix().eulerAngles(0, 1, 2));
}

bool Keyframe::calculate3dPoints(cv::Mat &lframe, cv::Mat &rframe,
    std::vector<Eigen::Vector4d> &points3d)
{
  std::vector<uchar> status1, status2;
  std::vector<float> err;
  std::vector<cv::Point2f> lpoints, rpoints;
  lpoints.resize(recent_features_.size());
  rpoints.resize(recent_features_.size());

  if (recent_features_.empty())
  {
    ROS_ERROR("No points aaaaah 1");
    return false;
  }

  // Match between last frame and current frame
  cv::calcOpticalFlowPyrLK(
      recent_, lframe, recent_features_, lpoints, status1, err, winSize,
      3, termcrit, 0, 0.001);
  removeFeatures(lpoints, rpoints, status1);
  
  if (recent_features_.empty())
  {
    ROS_ERROR("No points aaaaah 2");
    return false;
  }

  // Match between current left and right frame
  cv::calcOpticalFlowPyrLK(
      lframe, rframe, lpoints, rpoints, status2, err, winSize, 3,
      termcrit, 0, 0.001);
  removeFeatures(lpoints, rpoints, status2);

  if (recent_features_.empty())
  {
    ROS_ERROR("No points aaaaah 3");
    return false;
  }

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
  return true;
}

Eigen::Matrix4d Keyframe::getRelativePose(cv::Mat &lframe,
    cv::Mat& rframe, Eigen::Quaterniond orientation)
{
  keyframe_ok_ = calculate3dPoints(lframe, rframe, recent3d_);
  return getPoseDiffImageSpace(corrected_features_, recent3d_,
      orientation);
}


struct PlaneFitResidual {
  PlaneFitResidual(Eigen::Vector4d p) : p_(p) {}
  template <typename T> bool operator()(const T* const plane,
                                        T* residual) const
  {
    if (plane[0] == T(0) && plane[1] == T(0) && plane[2] == T(0))
    {
      return false;
    }
    residual[0] =
      (plane[0] * T(p_(0)) + plane[1] * T(p_(1)) +
       plane[2] * T(p_(2)) + plane[3]) / sqrt(plane[0] * plane[0] +
       plane[1] * plane[1] + plane[2] * plane[2]);
    residual[0] = residual[0] * residual[0];
    return true;
  }
 private:
  const Eigen::Vector4d p_;
};

Eigen::Matrix4d Keyframe::getGroundRelativePose()
{
  ceres::Problem problem;
  double plane[4] = {0, 0, -1, 0};
  for (int i = 0; i < recent3d_.size(); ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<PlaneFitResidual, 1, 4>(
          new PlaneFitResidual(recent3d_[i])),
        new ceres::HuberLoss(1.0), plane);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  ROS_WARN_STREAM("Summary: " << summary.BriefReport());
  ROS_ERROR_STREAM("Plane" << plane[0] << " " << plane[1] << " " <<
      plane[2] << " " << plane[3]);

  double rx = atan2(plane[0], plane[2]);
  double ry = atan2(plane[1], plane[2]);
  ROS_ERROR_STREAM("Plane rotation " << rx << " " << ry);
  Eigen::Affine3d r(create_rotation_matrix(rx, ry, 0));

  double d = plane[3] /
    (sqrt(pow(plane[0], 2) + pow(plane[1], 2) + pow(plane[2], 2)));
  Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(0, 0, fabs(d))));
  Eigen::Matrix4d pose = (t * r).matrix();
  return pose;
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

void debugImu(
    const Eigen::Quaternion<double>& e,
    const Eigen::Quaternion<double>& m,
    const Eigen::Quaternion<double>& d,
    const double& x, const double& y, const double& z)
{
  ROS_INFO_STREAM("got errors: " << x << " " << y << " " << z);
}

template<typename T> void debugImu(
    const Eigen::Quaternion<T>& e,
    const Eigen::Quaternion<T>& m,
    const Eigen::Quaternion<T>& d,
    const T& x, const T& y, const T& z)
{
  ROS_INFO_STREAM("got e " << e.w().a << " " << e.x().a << " " << e.y().a << " " << e.z().a);
  ROS_INFO_STREAM("got m " << m.w().a << " " << m.x().a << " " << m.y().a << " " << m.z().a);
  ROS_INFO_STREAM("got d " << d.w().a << " " << d.x().a << " " << d.y().a << " " << d.z().a);
  ROS_INFO_STREAM("got errors: " << x.a << " " << y.a << " " << z.a);
}

struct ImuResidual {
  ImuResidual(Eigen::Quaterniond rotation, double std_dev) :
    rotation_(rotation), std_dev_(std_dev) {}

  template<typename T> bool operator()(const T* const rotation,
      T* residual) const
  {
    Eigen::Quaternion<T> estimate =
        Eigen::Map<const Eigen::Quaternion<T> >(rotation);

    Eigen::Quaternion<T> measured(T(rotation_.w()), T(rotation_.x()),
        T(rotation_.y()), T(rotation_.z()));

    Eigen::Quaternion<T> diff = estimate * measured.conjugate();
    Eigen::Quaternion<T> half = diff; //(diff.matrix() / T(2)); 
    T q[4];
    q[0] = half.w();
    q[1] = half.x();
    q[2] = half.y();
    q[3] = half.z();

    ceres::QuaternionToAngleAxis(q, residual);  
    return true;
  }
private:
  Eigen::Quaterniond rotation_;
  double std_dev_;
};

struct ImageDistResidual {
  ImageDistResidual(cv::Point2f orig, Eigen::Vector4d p,
      CameraModel::CameraParams cparams, double std_dev, double weight)
      : orig_(orig), p_(p), cparams_(cparams), std_dev_(std_dev),
      weight_(weight)
  {
  }

  template <typename T> bool operator()(const T* const rotation,
                                        const T* const translation,
                                        T* residual) const
  {
    Eigen::Matrix<T,3,1> point;
    point << T(p_(0)), T(p_(1)), T(p_(2));

    // Map the T* array to an Eigen Quaternion object
    Eigen::Quaternion<T> q =
        Eigen::Map<const Eigen::Quaternion<T> >(rotation);

    Eigen::Matrix<T,3,1> t =
        Eigen::Map<const Eigen::Matrix<T,3,1> >(translation);
   
    // Transform point
    Eigen::Matrix<T,3,1> p = (q.matrix() / T(2)) * point;
    p += t;

    // Convert back into 2d space
    T xl = ((-p[1] * T(cparams_.f)) / p[0]) + T(cparams_.cx);
    T yl = ((-p[2] * T(cparams_.f)) / p[0]) + T(cparams_.cy);

    // Calculate the error between reprojected points
    T ex = (xl - T(orig_.x)) / T(std_dev_);
    T ey = (yl - T(orig_.y)) / T(std_dev_);
    residual[0] = (ex * ex) * weight_;
    residual[1] = (ey * ey) * weight_;
    return true;
  }

private:
  const cv::Point2f orig_;
  const Eigen::Vector4d p_;
  const CameraModel::CameraParams cparams_;
  const double std_dev_;
  const double weight_;
};

Eigen::Matrix4d Keyframe::getPoseDiffImageSpace(
    std::vector<cv::Point2f>& prevPoints,
    std::vector<Eigen::Vector4d>& currPoints,
    Eigen::Quaterniond orientation)
{
  if (!keyframe_ok_ || currPoints.empty())
  {
    is_lost_ = true;
    return Eigen::Matrix4d::Identity();
  }

  ceres::Problem problem;
  CameraModel::CameraParams cp = camera_model_.getAverageCamera();

  // TODO: Test if providing a better estimate helps
  // e.g. last cycle's estimate
  double rotation[4] = {0, 0, 0, 1}; // x, y, z, w
  double translation[3] = {0, 0, 0};

  ceres::LocalParameterization *quaternion_parameterization =
      new ceres_ext::EigenQuaternionParameterization;
      //new ceres::QuaternionParameterization;
  for (int i = 0; i < currPoints.size(); ++i) {
    // Give all points a std_dev of 2 pixels for now
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ImageDistResidual, 2, 4, 3>(
          new ImageDistResidual(prevPoints[i], currPoints[i], cp, 2, 0.005)),
        new ceres::HuberLoss(1.0), rotation, translation);
  }

  // Add imu residuals
  Eigen::Quaterniond d = keyframe_orientation_.inverse() * orientation;

  ROS_ERROR("Imu target is %lf %lf %lf %lf", orientation.w(),
      orientation.x(), orientation.y(), orientation.z());
  ROS_ERROR("Imu keyframe is %lf %lf %lf %lf", keyframe_orientation_.w(),
      keyframe_orientation_.x(), keyframe_orientation_.y(),
      keyframe_orientation_.z());
  ROS_ERROR("Imu diff is %lf %lf %lf %lf", d.w(), d.x(), d.y(), d.z());

  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<ImuResidual, 3, 4>(
          new ImuResidual(d, 1)), NULL, rotation);
  // Tell the solver rotation is a quaternion
  problem.SetParameterization(rotation, quaternion_parameterization);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  ROS_WARN_STREAM("Summary: " << summary.BriefReport());

  Eigen::Quaterniond qrot(rotation[3], rotation[0], rotation[1],
      rotation[2]);
  Eigen::Vector3d rpy = qrot.toRotationMatrix().eulerAngles(0, 1, 2);
  rpy *= 180.0 / M_PI;

  ROS_ERROR("Rotation %lf %lf %lf %lf",
      qrot.w(), qrot.x(), qrot.y(), qrot.z());
  ROS_ERROR_STREAM("Translation" << translation[0] << " "
      << translation[1] << " " << translation[2]);

  if (translation[0] > 2.0 || translation[1] > 2.0 ||
      translation[2] > 2.0)
  {
    ROS_ERROR("Solver fucked up");
    is_lost_ = true;
    return Eigen::Matrix4d::Identity();
  }

  if (!summary.IsSolutionUsable())
  {
    ROS_ERROR("Solver fail");
    abort();
    return Eigen::Matrix4d::Identity();
  }

  ROS_WARN_STREAM("convergance type " << summary.termination_type);

  Eigen::Affine3d r(qrot.toRotationMatrix());
  Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(translation[0],
          translation[1], translation[2])));
  Eigen::Matrix4d pose = (t * r).matrix();
  
  is_lost_ = false;
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


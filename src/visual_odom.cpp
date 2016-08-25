#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
#include <Eigen/Geometry> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>
#include <visual_odom/visual_odom.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace message_filters;

VisualOdom::VisualOdom(ros::NodeHandle &nh) :
    needToInit(true),
    termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03),
    subPixWinSize(10,10),
    winSize(31,31),
    max_count(100),
    left_sub(nh, "/usb_cam/left/image_raw", 1),
    right_sub(nh, "/usb_cam/right/image_raw", 1),
    sync(ImageSyncPolicy(10), left_sub, right_sub)
{
  sync.registerCallback(
      boost::bind(&VisualOdom::callback, this, _1, _2));

  cloudPub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("cloud", 1);
  debugCloudPub =
      nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("debug_cloud", 1);
  keyframeCloudPub =
      nh.advertise<pcl::PointCloud<pcl::PointXYZ> > ("keyframe_cloud", 1);

  CameraParams lp = {-0.169477, 0.0221934, 357.027, 246.735, 699.395, 0.12,
      0, 0, 0};
  l_cam_params = lp;
  CameraParams rp = {-0.170306, 0.0233104, 319.099, 218.565, 700.642, 0.12,
      -0.0166458, 0.0119791, -0.00187882};
  r_cam_params = rp;
}

Eigen::Matrix3d VisualOdom::create_rotation_matrix(double ax, double ay, double az) {
  Eigen::Matrix3d rx =
      Eigen::Matrix3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Matrix3d ry =
      Eigen::Matrix3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Matrix3d rz =
      Eigen::Matrix3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}

void VisualOdom::findPlotPoints(cv::Mat &grey, cv::Mat &prevGrey,
    std::vector<cv::Point2f> points, cv::Mat &frame, cv::Scalar color,
    std::vector<cv::Point2f> &output, std::vector<bool> &errors)
{
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(
      prevGrey, grey, points, output, status, err, winSize, 3,
      termcrit, 0, 0.001);
  
  size_t i, k;
  for (i = k = 0; i < output.size(); i++)
  {
    errors[i] = !status[i] | errors[i];
    if (!status[i])
    {  
      cv::circle(frame, output[i], 3, cv::Scalar(0, 0, 255), -1, 8);
      cv::line(frame, output[i], points[i], cv::Scalar(0, 0, 255),
          1, 8, 0);
    }
    else
    {
      cv::circle(frame, output[i], 3, color, -1, 8);
      cv::line(frame, output[i], points[i], color, 1, 8, 0);
    }
  }
  output.resize(i);
}

void VisualOdom::calculate3dPoints(std::vector<Eigen::Vector4d> &points3d,
    std::vector<cv::Point2f> points2d[], cv::Point2f midpoint)
{
  // TODO Take into account different principal points
  for (int i = 0; i < points2d[0].size(); ++i)
  {
    cv::Point2f l = points2d[0][i];
    cv::Point2f r = points2d[1][i];
    Eigen::Vector4d p;
    p(0) = 84.00222 / (l.x - r.x);
    p(1) = -((l.x - midpoint.x) * p(0)) / 700.0185; 
    p(2) = -((l.y - midpoint.y) * p(0)) / 700.0185;
    p(3) = 1;
    points3d.push_back(p);
  }
}

void VisualOdom::publishPointCloud(std::vector<Eigen::Vector4d> &points,
    std::string frame, ros::Publisher &pub)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (int i = 0; i < points.size(); ++i)
  {
    pcl::PointXYZ p(points[i](0), points[i](1), points[i](2));
    cloud.push_back(p);
  }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame;
  cloud.header.stamp = ros::Time::now().toSec();
  static int i = 0;
  cloud.header.seq = i++;
  pub.publish(cloud);
}

void VisualOdom::correctRadial(std::vector<cv::Point2f> &points,
    double k1, double k2, double cx, double cy, double fx, double fy,
    cv::Scalar color, cv::Mat &frame)
{
  for (int i = 0; i < points.size(); ++i)
  {
    double x = (points[i].x - cx) / fx;
    double y = (points[i].y - cy) / fy;
    double r2 = x * x + y * y;
    points[i].x = (x * (1 + k1 * r2 + k2 * r2 * r2)) * fx + cx;
    points[i].y = (y * (1 + k1 * r2 + k2 * r2 * r2)) * fy + cy;
    cv::circle(frame, points[i], 3, color, -1, 8);
  }
}

void VisualOdom::correctRotation(std::vector<cv::Point2f> &points, cv::Scalar color, cv::Mat &frame)
{
  Eigen::Matrix3d rotationMat = create_rotation_matrix(0.0166458, -0.0119791, 0.00187882);
  for (int i = 0; i < points.size(); ++i)
  {
    Eigen::Vector3d p(points[i].x, points[i].y, 700.642);
    p = rotationMat * p;
    points[i].x = p(0);
    points[i].y = p(1);
    cv::circle(frame, points[i], 3, color, -1, 8);
  }
}

void VisualOdom::detectBadMatches(std::vector<cv::Point2f> &lp,
    std::vector<cv::Point2f> &rp, std::vector<bool> &errors)
{
  for (int i = 0; i < lp.size(); ++i)
  {
    // If the points are not approximatly parallel then they cannot
    // be a good feature point match.
    // Points need to be corrected for any distortion for this to work.
    errors[i] = fabs(lp[i].y - rp[i].y) > 2 | errors[i];
  }
}


struct PointDistResidual {
  PointDistResidual(Eigen::Vector4d a, Eigen::Vector4d b)
      : a_(a), b_(b) {}
  template <typename T> bool operator()(const T* const rotation,
                                        const T* const translation,
                                        T* residual) const
  {
    // Rotate a by the passed in rotation and translation
    T R[9];
    ceres::EulerAnglesToRotationMatrix<T>(rotation, 3, R);
    T x = R[0] * T(a_(0)) + R[1] * T(a_(1)) + R[2]* T(a_(2)) - translation[0];
    T y = R[3] * T(a_(0)) + R[4] * T(a_(1)) + R[5]* T(a_(2)) - translation[1];
    T z = R[6] * T(a_(0)) + R[7] * T(a_(1)) + R[8]* T(a_(2)) - translation[2];
    x = x - T(b_(0));
    y = y - T(b_(1));
    z = z - T(b_(2));
    residual[0] = x * x;
    residual[1] = y * y;
    residual[2] = z * z;
    return true;
  }
 private:
  const Eigen::Vector4d a_;
  const Eigen::Vector4d b_;
};

struct PointDistRotResidual {
  PointDistRotResidual(Eigen::Vector4d a, Eigen::Vector4d b)
      : a_(a), b_(b) {}
  template <typename T> bool operator()(const T* const rotation,
                                        T* residual) const
  {
    // Rotate a by the passed in rotation and translation
    T R[9];
    ceres::EulerAnglesToRotationMatrix<T>(rotation, 3, R);
    T x = R[0] * T(a_(0)) + R[1] * T(a_(1)) + R[2]* T(a_(2));
    T y = R[3] * T(a_(0)) + R[4] * T(a_(1)) + R[5]* T(a_(2));
    T z = R[6] * T(a_(0)) + R[7] * T(a_(1)) + R[8]* T(a_(2));
    x = x - T(b_(0));
    y = y - T(b_(1));
    z = z - T(b_(2));
    residual[0] = x * x;
    residual[1] = y * y;
    residual[2] = z * z;
    return true;
  }
 private:
  const Eigen::Vector4d a_;
  const Eigen::Vector4d b_;
};

struct PointDistTransResidual {
  PointDistTransResidual(Eigen::Vector4d a, Eigen::Vector4d b)
      : a_(a), b_(b) {}
  template <typename T> bool operator()(const T* const translation,
                                        T* residual) const
  {
    // Rotate a by the passed in rotation and translation
    T x = T(a_(0)) - translation[0];
    T y = T(a_(1)) - translation[1];
    T z = T(a_(2)) - translation[2];
    x = x - T(b_(0));
    y = y - T(b_(1));
    z = z - T(b_(2));
    residual[0] = x * x;
    residual[1] = y * y;
    residual[2] = z * z;
    return true;
  }
 private:
  const Eigen::Vector4d a_;
  const Eigen::Vector4d b_;
};

Eigen::Matrix4d VisualOdom::getPoseDiff(
    std::vector<Eigen::Vector4d> &currPoints,
    std::vector<Eigen::Vector4d> &keyframePoints)
{
  ceres::Problem problem;
  double rotation[3] = {0, 0, 0};
  double translation[3] = {0, 0, 0};
  for (int i = 0; i < currPoints.size(); ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<PointDistResidual, 3, 3, 3>(new PointDistResidual(currPoints[i], keyframePoints[i])),
        new ceres::HuberLoss(1.0), rotation, translation);
  }

  /*for (int i = 0; i < currPoints.size(); ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<PointDistRotResidual, 3, 3>(new PointDistRotResidual(currPoints[i], keyframePoints[i])),
        new ceres::HuberLoss(1.0), rotation);
  }*/
  /*for (int i = 0; i < currPoints.size(); ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<PointDistTransResidual, 3, 3>(new PointDistTransResidual(currPoints[i], keyframePoints[i])),
        new ceres::HuberLoss(1.0), translation);
  }*/

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  ROS_WARN_STREAM("Summary: " << summary.BriefReport());
  ROS_ERROR_STREAM("Rotation" << rotation[0] << " " << rotation[1] << " " << rotation[2]);
  ROS_ERROR_STREAM("Translation" << translation[0] << " " << translation[1] << " " << translation[2]);
  Eigen::Affine3d r(create_rotation_matrix(rotation[0] * (M_PI / 180.0), rotation[1] * (M_PI / 180.0), rotation[2] * (M_PI / 180.0)));
  Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(-translation[0], -translation[1], -translation[2])));
  Eigen::Matrix4d pose = (t * r).matrix();
  return pose;
}
 
void VisualOdom::callback(const sensor_msgs::ImageConstPtr& left_image,
    const sensor_msgs::ImageConstPtr& right_image)
{
  cv::Mat frame_left = cv_bridge::toCvShare(
      left_image, sensor_msgs::image_encodings::BGR8)->image;
  cv::Mat frame_right = cv_bridge::toCvShare(
      right_image, sensor_msgs::image_encodings::BGR8)->image;

  cv::namedWindow("Left debug", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Right debug", cv::WINDOW_AUTOSIZE);

  cv::Mat lgrey, rgrey;
  cv::cvtColor(frame_left, lgrey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame_right, rgrey, cv::COLOR_BGR2GRAY);

  // needToInit = lpoints.size() < max_count / 4 || time expired;

  if (needToInit)
  {
    cv::goodFeaturesToTrack(lgrey, lpoints, max_count, 0.01, 10,
        cv::Mat(), 3, 0, 0.04);
    cv::cornerSubPix(lgrey, lpoints, subPixWinSize, cv::Size(-1,-1),
        termcrit);
    lgrey.copyTo(prevLGrey);
    lrpoints[0].resize(lpoints.size());
    lrpoints[1].resize(lpoints.size());
  }
  else if(!lpoints.empty())
  {
    std::vector<bool> errors;
    errors.resize(lpoints.size());
    findPlotPoints(lgrey, prevLGrey, lpoints, frame_left,
        cv::Scalar(0, 255, 0), lrpoints[0], errors);
    findPlotPoints(rgrey, lgrey, lrpoints[0], frame_right,
        cv::Scalar(0, 255, 0), lrpoints[1], errors);
    lpoints = lrpoints[0];
    cv::swap(prevLGrey, lgrey);

    // Need to correct for rotation between camera and radial distortion
    correctRadial(lrpoints[0], -0.169477, 0.0221934, 357.027, 246.735,
        699.395, 699.395, cv::Scalar(255, 0, 255), frame_right);
    correctRadial(lrpoints[1], -0.170306, 0.0233104, 319.099, 218.565,
        700.642, 700.642, cv::Scalar(255, 0, 255), frame_right);
    correctRotation(lrpoints[0], cv::Scalar(255, 0, 0), frame_right);

    detectBadMatches(lrpoints[0], lrpoints[1], errors); 

    static std::vector<Eigen::Vector4d> prevPoints3d;
    static bool pointsInit = false;
    for (int i = lpoints.size() - 1; i >= 0; --i)
    {
      if (errors[i])
      {
        lpoints.erase(lpoints.begin() + i);
        lrpoints[0].erase(lrpoints[0].begin() + i);
        lrpoints[1].erase(lrpoints[1].begin() + i);
        if (pointsInit)
        {
          prevPoints3d.erase(prevPoints3d.begin() + i);
        }
      }
    }

    // Calculate 3d points
    cv::Point2d pp(338.063, 232.65);
    std::vector<Eigen::Vector4d> points3d, tPoints3d;
    calculate3dPoints(points3d, lrpoints, pp);
    publishPointCloud(points3d, "camera", cloudPub);
    publishPointCloud(prevPoints3d, "camera", keyframeCloudPub);
    if (!pointsInit)
    {
      prevPoints3d = points3d;
      pointsInit = true;
    }

    Eigen::Matrix4d pose = getPoseDiff(points3d, prevPoints3d);

    // Rotate points to align to key frame for debug
    for (int i = 0; i < points3d.size(); ++i)
    {
      tPoints3d.push_back(pose * points3d[i]);
    }
    publishPointCloud(tPoints3d, "camera", debugCloudPub);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(-pose(0, 3), -pose(1, 3), -pose(2, 3)));
    //transform.setOrigin(tf::Vector3(0, 0, 0));
    tf::Quaternion q;
    Eigen::Matrix3d rotMat = pose.block<3, 3>(0, 0);
    Eigen::Vector3d rot = rotMat.eulerAngles(0, 1, 2); 
    q.setRPY(rot(0), rot(1), rot(2));
    //q.setRPY(0, 0, 0);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(
          transform, ros::Time::now(), "map", "camera")); 

  }

  needToInit = false;
  cv::imshow("Left debug", frame_left);
  cv::imshow("Right debug", frame_right);
  cv::waitKey(1);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "visual_node");
  ros::NodeHandle nh;
  VisualOdom vo(nh);
  ROS_INFO("Starting visual odom");
  ros::spin();
  return 0;
}

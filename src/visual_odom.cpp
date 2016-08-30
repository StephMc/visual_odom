#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
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
    max_count(200),
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
  //CameraParams lp = {-0.173774, 0.0262478, 663.473, 351.115, 699.277, 0.12,
  //    0, 0, 0};
  l_cam_params = lp;
  CameraParams rp = {-0.170306, 0.0233104, 319.099, 218.565, 700.642, 0.12,
      -0.0166458, 0.0119791, -0.00187882};
  //CameraParams rp = {-0.172575, 0.0255858, 673.393, 349.306, 700.72, 0.12,
  //    -0.00251904, 0.0139689, 0.000205762};
  r_cam_params = rp;
  //basePose = Eigen::Matrix4d::Identity();
  //pose = Eigen::Matrix4d::Identity();
  CameraPose zero = {0, 0, 0, 0, 0, 0};
  currPose = zero;
  basePose = zero;
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
  double focal = (l_cam_params.f + r_cam_params.f) / 2;
  double fb = focal * r_cam_params.b;
  for (int i = 0; i < points2d[0].size(); ++i)
  {
    cv::Point2f l = points2d[0][i];
    cv::Point2f r = points2d[1][i];
    Eigen::Vector4d p;
    p(0) = fb / (l.x - r.x);
    p(1) = -((l.x - midpoint.x) * p(0)) / focal; 
    p(2) = -((l.y - midpoint.y) * p(0)) / focal;
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
  pcl_conversions::toPCL(ros::Time::now(), cloud.header.stamp);
  //cloud.header.stamp = ros::Time::now().toSec();
  static int i = 0;
  cloud.header.seq = i++;
  pub.publish(cloud);
}

void VisualOdom::correctRadial(std::vector<cv::Point2f> &points,
    double k1, double k2, double cx, double cy, double f,
    cv::Scalar color, cv::Mat &frame)
{
  for (int i = 0; i < points.size(); ++i)
  {
    double x = (points[i].x - cx) / f;
    double y = (points[i].y - cy) / f;
    double r2 = x * x + y * y;
    points[i].x = (x * (1 + k1 * r2 + k2 * r2 * r2)) * f + cx;
    points[i].y = (y * (1 + k1 * r2 + k2 * r2 * r2)) * f + cy;
    cv::circle(frame, points[i], 3, color, -1, 8);
  }
}

void VisualOdom::correctRotation(std::vector<cv::Point2f> &points, cv::Scalar color, cv::Mat &frame)
{
  Eigen::Matrix3d rotationMat = create_rotation_matrix(-r_cam_params.rx,
      -r_cam_params.ry, -r_cam_params.rz);
  for (int i = 0; i < points.size(); ++i)
  {
    Eigen::Vector3d p(points[i].x, points[i].y, r_cam_params.f);
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
    errors[i] = fabs(lp[i].y - rp[i].y) > 3 | errors[i];
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
 
struct ImageDistResidual {
  ImageDistResidual(cv::Point2f orig, Eigen::Vector4d p,
      VisualOdom::CameraParams cparams)
      : orig_(orig), p_(p), cparams_(cparams) {}
  template <typename T> bool operator()(const T* const rotation,
                                        const T* const translation,
                                        T* residual) const
  {
    // Rotate a by the passed in rotation and translation
    T R[9];
    ceres::EulerAnglesToRotationMatrix<T>(rotation, 3, R);
    T x = R[0] * T(p_(0)) + R[1] * T(p_(1)) + R[2]* T(p_(2)) - translation[0];
    T y = R[3] * T(p_(0)) + R[4] * T(p_(1)) + R[5]* T(p_(2)) - translation[1];
    T z = R[6] * T(p_(0)) + R[7] * T(p_(1)) + R[8]* T(p_(2)) - translation[2];
   
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
  const VisualOdom::CameraParams cparams_;
};

Eigen::Matrix4d VisualOdom::getPoseDiffImageSpace(
    std::vector<cv::Point2f> prevPoints,
    std::vector<Eigen::Vector4d> currPoints)
{
  ceres::Problem problem;
  CameraParams cp = {0, 0,
      (l_cam_params.cx + r_cam_params.cx) / 2,
      (l_cam_params.cy + r_cam_params.cy) / 2,
      (l_cam_params.f + r_cam_params.f) / 2,
      0, 0, 0, 0};
  double rotation[3] = {
    currPose.rx * (180.0 / M_PI),
    currPose.ry * (180.0 / M_PI),
    currPose.rz * (180.0 / M_PI)};
  double translation[3] = {currPose.x, currPose.y, currPose.z};
  for (int i = 0; i < currPoints.size(); ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ImageDistResidual, 2, 3, 3>(new ImageDistResidual(prevPoints[i], currPoints[i], cp)),
        new ceres::HuberLoss(1.0), rotation, translation);
  }

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

  static bool pointsInit = false;
  static ros::Time prev = ros::Time(0);
  //needToInit = (ros::Time::now() - prev).toSec() > 1;
  if (needToInit)
  {
    prev = ros::Time::now();
    ROS_ERROR("New points!!!");
    pointsInit = false;
    //basePose = pose;
    cv::goodFeaturesToTrack(lgrey, opoints, max_count, 0.01, 10,
        cv::Mat(), 3, 0, 0.04);
    cv::cornerSubPix(lgrey, opoints, subPixWinSize, cv::Size(-1,-1),
        termcrit);
    lpoints = opoints;
    lgrey.copyTo(prevLGrey);
    lrpoints[0].resize(lpoints.size());
    lrpoints[1].resize(lpoints.size());
  } 
 
  // TODO: Handle complete loss of lpoints
  if (!lpoints.empty())
  {
    std::vector<bool> errors;
    errors.resize(lpoints.size());
    findPlotPoints(lgrey, prevLGrey, lpoints, frame_left,
        cv::Scalar(0, 255, 0), lrpoints[0], errors);
    findPlotPoints(rgrey, lgrey, lrpoints[0], frame_right,
        cv::Scalar(0, 255, 0), lrpoints[1], errors);
    lpoints = lrpoints[0];
    lgrey.copyTo(prevLGrey);

    // Need to correct for rotation between camera and radial distortion
    correctRadial(lrpoints[0],
        l_cam_params.k1, l_cam_params.k2,
        l_cam_params.cx, l_cam_params.cy,
        l_cam_params.f, cv::Scalar(255, 0, 255), frame_left);
    correctRadial(lrpoints[1],
        r_cam_params.k1, r_cam_params.k2,
        r_cam_params.cx, r_cam_params.cy,
        r_cam_params.f, cv::Scalar(255, 0, 255), frame_right);
    correctRotation(lrpoints[0], cv::Scalar(255, 0, 0), frame_right);
    detectBadMatches(lrpoints[0], lrpoints[1], errors); 

    for (int i = lpoints.size() - 1; i >= 0; --i)
    {
      if (errors[i])
      {
        lpoints.erase(lpoints.begin() + i);
        opoints.erase(opoints.begin() + i);
        lrpoints[0].erase(lrpoints[0].begin() + i);
        lrpoints[1].erase(lrpoints[1].begin() + i);
        if (pointsInit)
        {
          prevPoints3d.erase(prevPoints3d.begin() + i);
        }
      }
    }

    // Calculate 3d points
    cv::Point2d pp((l_cam_params.cx + r_cam_params.cx) / 2,
        (l_cam_params.cy + r_cam_params.cy) / 2);
    std::vector<Eigen::Vector4d> points3d, tPoints3d;
    calculate3dPoints(points3d, lrpoints, pp);
    publishPointCloud(points3d, "camera", cloudPub);
    publishPointCloud(prevPoints3d, "map", keyframeCloudPub);
    if (!pointsInit)
    {
      prevPoints3d = points3d;
      pointsInit = true;
    }

    //pose = getPoseDiff(points3d, prevPoints3d);
    Eigen::Matrix4d pose = getPoseDiffImageSpace(opoints, points3d);

    // Rotate points to align to key frame for debug
    for (int i = 0; i < points3d.size(); ++i)
    {
      tPoints3d.push_back(pose * points3d[i]);
    }
    publishPointCloud(tPoints3d, "map", debugCloudPub);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    currPose.x = pose(0, 3);
    currPose.y = pose(1, 3);
    currPose.z = pose(2, 3);
    transform.setOrigin(tf::Vector3(pose(0, 3), pose(1, 3), pose(2, 3)));
    tf::Quaternion q;
    Eigen::Matrix3d rotMat = pose.block<3, 3>(0, 0);
    Eigen::Vector3d rot = rotMat.eulerAngles(0, 1, 2); 
    q.setRPY(rot(0), rot(1), rot(2));
    currPose.rx = rot(0);
    currPose.ry = rot(1);
    currPose.rz = rot(2);
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

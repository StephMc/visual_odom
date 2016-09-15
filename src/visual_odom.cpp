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
#include <nav_msgs/Odometry.h>
#include <boost/assign.hpp>

using namespace message_filters;

VisualOdom::VisualOdom(ros::NodeHandle &nh) :
    need_new_keyframe_(true),
    left_sub_(nh, "/camera/left/image_raw", 1),
    //left_sub_(nh, "/usb_cam/left/image_raw", 1),
    right_sub_(nh, "/camera/right/image_raw", 1),
    //right_sub_(nh, "/usb_cam/right/image_raw", 1),
    sync_(ImageSyncPolicy(10), left_sub_, right_sub_),
    curr_keyframe_(NULL), prev_keyframe_(NULL),
    camera_model_(
        //CameraModel::CameraParams(-0.169477, 0.0221934, 357.027, 246.735,
        //    699.395, 0.12, 0, 0, 0),
        //CameraModel::CameraParams(-0.170306, 0.0233104, 319.099, 218.565,
        //    700.642, 0.12, 0.0166458, 0.0119791, 0.00187882)
        CameraModel::CameraParams(-0.173774, 0.0262478, 343.473, 231.115,
          699.277, 0.12, 0, 0, 0),
        CameraModel::CameraParams(-0.172575, 0.0255858, 353.393, 229.306,
          700.72, 0.12, -0.00251904, 0.0139689, 0.000205762))
{
  // Fetch config parameters
  nh.param("max_feature_count", max_feature_count_, 50);
  std::string left_image_topic, right_image_topic;
  nh.param("left_image_topic", left_image_topic,
      std::string("/usb_cam/left/image_raw"));
  nh.param("right_image_topic", right_image_topic,
      std::string("/usb_cam/right/image_raw"));

  // Initialize start pose
  keyframe_pose_ = Eigen::Matrix4d::Identity();

  // Setup image callback
  sync_.registerCallback(
      boost::bind(&VisualOdom::callback, this, _1, _2));

  // Setup cloud pubishers
  cloud_pub_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("cloud", 1);
  debug_cloud_pub_ =
      nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("debug_cloud", 1);
  keyframe_cloud_pub_ =
      nh.advertise<pcl::PointCloud<pcl::PointXYZ> > ("keyframe_cloud", 1);

  // Setup odometry publisher
  odom_pub_ = nh.advertise<nav_msgs::Odometry>("vo", 1);
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
  static int i = 0;
  cloud.header.seq = i++;
  pub.publish(cloud);
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

void VisualOdom::drawPoints(cv::Mat& frame,
    std::vector<cv::Point2f> points, cv::Scalar color)
{
  for (int i = 0; i < points.size(); ++i)
  {
    cv::circle(frame, points[i], 3, color, -1, 8);
  }
}

void VisualOdom::callback(const sensor_msgs::ImageConstPtr& left_image,
    const sensor_msgs::ImageConstPtr& right_image)
{
  // TODO don't copy here
  cv::Mat frame_left = cv_bridge::toCvShare(
      left_image, sensor_msgs::image_encodings::BGR8)->image;
  cv::Mat frame_right = cv_bridge::toCvShare(
      right_image, sensor_msgs::image_encodings::BGR8)->image;

  cv::Mat lgrey, rgrey;
  cv::cvtColor(frame_left, lgrey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame_right, rgrey, cv::COLOR_BGR2GRAY);

  if (curr_keyframe_ == NULL)
  {
    ROS_WARN("Initializing");
    need_new_keyframe_ = false;
    curr_keyframe_ = new Keyframe(lgrey, rgrey, camera_model_,
        max_feature_count_);
  } 

  // Publish keyframe transform
  tf::Transform ktransform;
  ktransform.setOrigin(
      tf::Vector3(keyframe_pose_(0, 3), keyframe_pose_(1, 3),
          keyframe_pose_(2, 3)));

  tf::Quaternion kq;
  Eigen::Matrix3d krotMat = keyframe_pose_.block<3, 3>(0, 0);
  Eigen::Vector3d krot = krotMat.eulerAngles(0, 1, 2); 
  kq.setRPY(krot(0), krot(1), krot(2));
  ktransform.setRotation(kq);

  br_.sendTransform(tf::StampedTransform(
      ktransform, ros::Time::now(), "map", "keyframe")); 

  // Get camera pose
  camera_pose_ = curr_keyframe_->getRelativePose(lgrey, rgrey);

  // Get pose relative to ground (uses latest solution to getRelativePose)
  Eigen::Matrix4d ground_relative_pose_ =
      curr_keyframe_->getGroundRelativePose();

  tf::Transform gtransform;
  gtransform.setOrigin(
      tf::Vector3(ground_relative_pose_(0, 3),
          ground_relative_pose_(1, 3),
          ground_relative_pose_(2, 3)));

  tf::Quaternion gq;
  Eigen::Matrix3d grotMat = ground_relative_pose_.block<3, 3>(0, 0);
  Eigen::Vector3d grot = grotMat.eulerAngles(0, 1, 2); 
  kq.setRPY(grot(0), grot(1), grot(2));
  gtransform.setRotation(gq);

  br_.sendTransform(tf::StampedTransform(
      gtransform, ros::Time::now(), "ground", "flappy")); 
 
  // For debug
  publishPointCloud(curr_keyframe_->getRecent3d(), "camera", cloud_pub_);
  publishPointCloud(curr_keyframe_->getKeyframe3d(), "keyframe",
      keyframe_cloud_pub_);

  // Rotate points to align to key frame for debug
  std::vector<Eigen::Vector4d> &points3d = curr_keyframe_->getRecent3d();
  std::vector<Eigen::Vector4d> t_points3d;
  for (int i = 0; i < points3d.size(); ++i)
  {
    t_points3d.push_back(camera_pose_ * points3d[i]);
  }
  publishPointCloud(t_points3d, "keyframe", debug_cloud_pub_);

  // Publish transform
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(camera_pose_(0, 3),
      camera_pose_(1, 3), camera_pose_(2, 3)));

  tf::Quaternion q;
  Eigen::Matrix3d rotMat = camera_pose_.block<3, 3>(0, 0);
  Eigen::Vector3d rot = rotMat.eulerAngles(0, 1, 2); 
  q.setRPY(rot(0), rot(1), rot(2));
  transform.setRotation(q);

  br_.sendTransform(tf::StampedTransform(
        transform, ros::Time::now(), "keyframe", "camera")); 

  // Publish odometry
  Eigen::Matrix4d current_pose = keyframe_pose_ * camera_pose_;
  nav_msgs::Odometry odom;
  static unsigned int seq_odom = 0;
  odom.header.seq = seq_odom++;
  odom.header.stamp = ros::Time::now();
  odom.header.frame_id = "map";
  odom.child_frame_id = "vo_child_frame";
  
  // Translation
  odom.pose.pose.position.x = current_pose(0, 3);
  odom.pose.pose.position.y = current_pose(1, 3);
  odom.pose.pose.position.z = current_pose(2, 3);
  
  // Rotation
  Eigen::Matrix3d crotMat = current_pose.block<3, 3>(0, 0);
  Eigen::Vector3d crot = crotMat.eulerAngles(0, 1, 2); 
  odom.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(
      crot(0), crot(1), crot(2));

  odom.pose.covariance = 
      boost::assign::list_of (1e-2) (0) (0)  (0)  (0)  (0)
                              (0) (1e-2)  (0)  (0)  (0)  (0)
                              (0)   (0)  (1e-2) (0)  (0)  (0)
                              (0)   (0)   (0) (1e-2) (0)  (0)
                              (0)   (0)   (0)  (0) (1e-2) (0)
                              (0)   (0)   (0)  (0)  (0)  (1e-2) ;

  odom_pub_.publish(odom);

  static Eigen::Matrix4d prev_pose_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d diff_pose = prev_pose_ * current_pose;
  prev_pose_ = current_pose;
  static ros::Time prev_time_ = ros::Time::now();
  double time_diff = (ros::Time::now() - prev_time_).toSec();
  prev_time_ = ros::Time::now();
  if (time_diff == 0) time_diff = 1;

  // Velocity
  odom.twist.twist.linear.x = diff_pose(0, 3) / time_diff; 
  odom.twist.twist.linear.y = diff_pose(1, 3) / time_diff; 
  odom.twist.twist.linear.z = diff_pose(2, 3) / time_diff; 

  // Angular velocity
  Eigen::Matrix3d vrotMat = diff_pose.block<3, 3>(0, 0);
  Eigen::Vector3d vrot = vrotMat.eulerAngles(0, 1, 2); 
  odom.twist.twist.angular.x = vrot(0) / time_diff; 
  odom.twist.twist.angular.y = vrot(1) / time_diff; 
  odom.twist.twist.angular.z = vrot(2) / time_diff; 

  odom.twist.covariance = 
      boost::assign::list_of (1e-2) (0) (0)  (0)  (0)  (0)
                              (0) (1e-2)  (0)  (0)  (0)  (0)
                              (0)   (0)  (1e-2) (0)  (0)  (0)
                              (0)   (0)   (0) (1e-2) (0)  (0)
                              (0)   (0)   (0)  (0) (1e-2) (0)
                              (0)   (0)   (0)  (0)  (0)  (1e-2) ;

  // Check if we're currently far enough away from the keyframe to
  // trigger getting a new keyframe
  double max_rad = 0.16; // ~5 degrees
  double rx = fabs(rot(0));
  double ry = fabs(rot(1));
  double rz = fabs(rot(2));
  rx = rx > (M_PI / 2) ? M_PI - rx : rx;
  ry = ry > (M_PI / 2) ? M_PI - ry : ry;
  rz = rz > (M_PI / 2) ? M_PI - rz : rz;
  ROS_WARN("got rot %lf, %lf, %lf", rx, ry, rz);
  if (rx > max_rad || ry > max_rad || rz > max_rad)
  {
    ROS_INFO("New keyframe by rotation");
    need_new_keyframe_ = true;
  }

  if (sqrt(pow(camera_pose_(0, 3), 2) + pow(camera_pose_(1, 3), 2) +
        pow(camera_pose_(2, 3), 2)) > 0.5)
  {
    ROS_INFO("New keyframe by translation");
    need_new_keyframe_ = true;
  }

  if (need_new_keyframe_)
  {
    ROS_WARN("Getting new keyframe");
    need_new_keyframe_ = false;
    if (prev_keyframe_ != NULL)
    {
      free(prev_keyframe_);
    }
    prev_keyframe_ = curr_keyframe_;
    curr_keyframe_ = new Keyframe(lgrey, rgrey, camera_model_,
        max_feature_count_);

    if (prev_keyframe_ != NULL)
    {
      // Add to last keyframe estimate
      keyframe_pose_ = keyframe_pose_ * camera_pose_;
    }
  } 
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

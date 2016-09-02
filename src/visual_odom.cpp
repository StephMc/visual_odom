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

using namespace message_filters;

VisualOdom::VisualOdom(ros::NodeHandle &nh) :
    need_new_keyframe_(true),
    left_sub_(nh, "/usb_cam/left/image_raw", 1),
    right_sub_(nh, "/usb_cam/right/image_raw", 1),
    sync_(ImageSyncPolicy(10), left_sub_, right_sub_),
    curr_keyframe_(NULL), prev_keyframe_(NULL),
    camera_model_(
        CameraModel::CameraParams(-0.169477, 0.0221934, 357.027, 246.735,
          699.395, 0.12, 0, 0, 0),
        CameraModel::CameraParams(-0.170306, 0.0233104, 319.099, 218.565,
          700.642, 0.12, 0.0166458, -0.0119791, 0.00187882))
{
  // Fetch config parameters
  nh.param("max_feature_count", max_feature_count_, 100);
  std::string left_image_topic, right_image_topic;
  nh.param("left_image_topic", left_image_topic,
      std::string("/usb_cam/left/image_raw"));
  nh.param("right_image_topic", right_image_topic,
      std::string("/usb_cam/right/image_raw"));

  // Setup image callback
  sync_.registerCallback(
      boost::bind(&VisualOdom::callback, this, _1, _2));

  // Setup cloud pubishers
  cloud_pub_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("cloud", 1);
  debug_cloud_pub_ =
      nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("debug_cloud", 1);
  keyframe_cloud_pub_ =
      nh.advertise<pcl::PointCloud<pcl::PointXYZ> > ("keyframe_cloud", 1);
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

  if (need_new_keyframe_)
  {
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
      // Calculate the transform between keyframes
    }
  } 

  Eigen::Matrix4d pose = curr_keyframe_->getRelativePose(lgrey, rgrey);
 
  // For debug
  publishPointCloud(curr_keyframe_->getRecent3d(), "camera", cloud_pub_);
  publishPointCloud(curr_keyframe_->getKeyframe3d(), "keyframe",
      keyframe_cloud_pub_);

  // Rotate points to align to key frame for debug
  std::vector<Eigen::Vector4d> &points3d = curr_keyframe_->getRecent3d();
  std::vector<Eigen::Vector4d> t_points3d;
  for (int i = 0; i < points3d.size(); ++i)
  {
    t_points3d.push_back(pose * points3d[i]);
  }
  publishPointCloud(t_points3d, "keyframe", debug_cloud_pub_);

  // Publish transform
  tf::Transform transform;
  transform.setOrigin(tf::Vector3(pose(0, 3), pose(1, 3), pose(2, 3)));

  tf::Quaternion q;
  Eigen::Matrix3d rotMat = pose.block<3, 3>(0, 0);
  Eigen::Vector3d rot = rotMat.eulerAngles(0, 1, 2); 
  q.setRPY(rot(0), rot(1), rot(2));
  transform.setRotation(q);

  br_.sendTransform(tf::StampedTransform(
        transform, ros::Time::now(), "keyframe", "camera")); 

  // Check if we're currently far enough away from the keyframe to
  // trigger getting a new keyframe
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

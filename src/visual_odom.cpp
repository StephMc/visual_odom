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

using namespace message_filters;

Eigen::Matrix3d create_rotation_matrix(double ax, double ay, double az) {
  Eigen::Matrix3d rx =
      Eigen::Matrix3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Matrix3d ry =
      Eigen::Matrix3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Matrix3d rz =
      Eigen::Matrix3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}

/*void getRotation(std::vector<cv::Point2f> points[], double focal, cv::Point2d pp)
{
  cv::Mat E, R, t, mask;
  E = cv::findEssentialMat(points[0], points[1], focal, pp, RANSAC, 0.999, 1.0, mask);
  cv::recoverPose(E, points[0], points[1], R, t, focal, pp, mask);
  cv::Point3f p;
  p.x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
  p.y = atan2(-R.at<double>(2, 0), sqrt(pow(R.at<double>(2, 1), 2) + pow(R.at<double>(2, 2), 2)));
  p.z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
  return p;
}*/

cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
cv::Size subPixWinSize(10,10), winSize(31,31);
int max_count = 100;

void findPlotPoints(cv::Mat &grey, cv::Mat &prevGrey,
    std::vector<cv::Point2f> points[], cv::Mat &frame, cv::Scalar color,
    std::vector<cv::Point2f> &output)
{
  std::vector<uchar> status;
  std::vector<float> err;
  if (prevGrey.empty())
  {
    abort(); //grey.copyTo(prevGrey);
  }
  cv::calcOpticalFlowPyrLK(
      //prevGrey, grey, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);
      prevGrey, grey, points[0], output, status, err, winSize, 3, termcrit, 0, 0.001);
  
  size_t i, k;
  for (i = k = 0; i < output.size(); i++)
  {
    //if (!status[i]) continue;
    //output[k++] = output[i];
    if (!status[i])
    {  
      cv::circle(frame, output[i], 3, cv::Scalar(0, 0, 255), -1, 8);
      cv::line(frame, output[i], points[0][i], cv::Scalar(0, 0, 255), 1, 8, 0);
    }
    else
    {
      cv::circle(frame, output[i], 3, color, -1, 8);
      cv::line(frame, output[i], points[0][i], color, 1, 8, 0);
    }
  }
  output.resize(i);
}

void calculate3dPoints(std::vector<Eigen::Vector3d> &points3d,
    std::vector<cv::Point2f> points2d[], cv::Point2f midpoint)
{
  for (int i = 0; i < points2d[0].size(); ++i)
  {
    cv::Point2f l = points2d[0][i];
    cv::Point2f r = points2d[1][i];
    Eigen::Vector3d p;
    p(0) = 84.0864 / (l.x - r.x);
    p(1) = -((l.x - midpoint.x) * p(0)) / 699.277; 
    //p(1) = -(atan((2 * (l.x - midpoint.x)) / 699.277) * p(0));
    p(2) = -((l.y - midpoint.y) * p(0)) / 699.277;
    //p(2) = -(atan((2 * (l.y - midpoint.y)) / 699.277) * p(0));
    points3d.push_back(p);
  }
}

ros::Publisher cloudPub, debugCloudPub, keyframeCloudPub;
void publishPointCloud(std::vector<Eigen::Vector3d> &points,
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

void correctRadial(std::vector<cv::Point2f> &points, double k1, double k2,
    double cx, double cy, double fx, double fy, cv::Scalar color, cv::Mat &frame)
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

void correctRotation(std::vector<cv::Point2f> &points, cv::Scalar color, cv::Mat &frame)
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

std::vector<cv::Point2f> lpoints[2], rpoints[2], lrpoints[2], rlpoints[2];
cv::Mat prevLGrey, prevRGrey;
bool needToInit = true;
void callback(const sensor_msgs::ImageConstPtr& left_image,
    const sensor_msgs::ImageConstPtr& right_image,
    const geometry_msgs::QuaternionStampedConstPtr& imu_rotation)
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

  if (needToInit)
  {
    cv::goodFeaturesToTrack(lgrey, lpoints[0], max_count, 0.01, 10, cv::Mat(), 3, 0, 0.04);
    cv::cornerSubPix(lgrey, lpoints[0], subPixWinSize, cv::Size(-1,-1), termcrit);

    cv::goodFeaturesToTrack(rgrey, rpoints[0], max_count, 0.01, 10, cv::Mat(), 3, 0, 0.04);
    cv::cornerSubPix(rgrey, rpoints[0], subPixWinSize, cv::Size(-1,-1), termcrit);
    lgrey.copyTo(prevLGrey);
    lrpoints[0].resize(lpoints[0].size());
    lrpoints[1].resize(lpoints[0].size());
    //lrpoints[1] = lpoints[1];
    //rlpoints[1] = rpoints[1];
  }
  else if(!lpoints[0].empty() && !rpoints[0].empty())
  {
    findPlotPoints(lgrey, prevLGrey, lpoints, frame_left, cv::Scalar(0, 255, 0), lrpoints[0]);
    //findPlotPoints(rgrey, prevRGrey, rpoints, frame_right, cv::Scalar(255, 0, 0));

    //lrpoints[0] = lpoints[1];
    //rlpoints[0] = rpoints[1];

    //findPlotPoints(lgrey, rgrey, rlpoints, frame_left, cv::Scalar(255, 0, 0));
    findPlotPoints(rgrey, lgrey, lrpoints, frame_right, cv::Scalar(0, 255, 0), lrpoints[1]);

    // Need to correct for rotation between camera and for radial distortion
    correctRadial(lrpoints[0], -0.169477, 0.0221934, 357.027, 246.735, 699.395, 699.395, cv::Scalar(255, 0, 255), frame_right);
    correctRadial(lrpoints[1], -0.170306, 0.0233104, 319.099, 218.565, 700.642, 700.642, cv::Scalar(255, 0, 255), frame_right);
    correctRotation(lrpoints[0], cv::Scalar(255, 0, 0), frame_right);

    // TODO Filter points based on stability, epipolar match

    // lpoints and rpoints contain the rotation information
    // rlpoints and lrpoints contain the 3d information for the current points
    // Need to compare the keyframe 3d points to the current 3d points
    // First apply the rotation then average the translation

    // Find rotation between frames
    double focal = 699.277; // TODO: Get this from camera info
    //cv::Point2d pp(frame_left.rows/2, frame_left.cols/2);
    cv::Point2d pp(357.027, 246.735);
    //cv::Point3f rot = getRotation(lpoints, focal, pp);
    cv::Point3d rot;
    static cv::Point3d initRot;
    tf::Quaternion quat;
    tf::quaternionMsgToTF(imu_rotation->quaternion, quat);
    double y, p, r;
    tf::Matrix3x3(quat).getEulerYPR(y, p, r);
    rot.z = y;
    rot.y = p;
    rot.x = r;
    static bool pointsInit = false;
    if (!pointsInit)
    {
      initRot = rot;
    }
    Eigen::Matrix3d rotationMat = create_rotation_matrix((rot.x - initRot.x)/2, (rot.y - initRot.y)/2, (rot.z - initRot.z)/2);
    //Eigen::Matrix3d rotationMat = create_rotation_matrix(0, 0, 0);

    // Calculate 3d points
    std::vector<Eigen::Vector3d> points3d, tPoints3d;
    static std::vector<Eigen::Vector3d> prevPoints3d;
    calculate3dPoints(points3d, lrpoints, pp);
    publishPointCloud(points3d, "camera", cloudPub);

    // Rotate points to align to key frame
    for (int i = 0; i < points3d.size(); ++i)
    {
      tPoints3d.push_back(rotationMat * points3d[i]);
    }
    publishPointCloud(tPoints3d, "camera", debugCloudPub);

    // Compare distance between current and keyframe points
    Eigen::Vector3d averageDist;
    for (int i = 0; i < prevPoints3d.size(); ++i)
    {
      averageDist += prevPoints3d[i] - tPoints3d[i];
    }
    averageDist /= points3d.size(); // TODO check for 0 points
    ROS_WARN_STREAM("Moved " << averageDist << " Used points " << points3d.size());
    if (!pointsInit)
    {
      prevPoints3d = points3d;
      pointsInit = true;
    }
    publishPointCloud(prevPoints3d, "camera", keyframeCloudPub);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(averageDist(0), averageDist(1), averageDist(2)));
    //transform.setOrigin(tf::Vector3(0, 0, 0));
    tf::Quaternion q;
    q.setRPY((initRot.x - rot.x)/2, (initRot.y - rot.y)/2, (initRot.z - rot.z)/2);
    //q.setRPY(0, 0, 0);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "camera")); 
  }

  needToInit = false;
  cv::imshow("Left debug", frame_left);
  cv::imshow("Right debug", frame_right);
  cv::waitKey(1);
  //std::swap(lpoints[1], lpoints[0]);
  //cv::swap(prevLGrey, lgrey);

  //std::swap(rpoints[1], rpoints[0]);
  //cv::swap(prevRGrey, rgrey);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "visual_node");
  ROS_INFO("Starting visual odom");
  ros::NodeHandle nh;

  message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/usb_cam/left/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/usb_cam/right/image_raw", 1);
  message_filters::Subscriber<geometry_msgs::QuaternionStamped> imu_sub(nh, "/orientation", 1);

  typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::QuaternionStamped> ImageSyncPolicy;
  Synchronizer<ImageSyncPolicy> sync(ImageSyncPolicy(10), left_sub, right_sub, imu_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));

  cloudPub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> > ("cloud", 1);
  debugCloudPub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> > ("debug_cloud", 1);
  keyframeCloudPub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> > ("keyframe_cloud", 1);
  ros::spin();

  return 0;
}

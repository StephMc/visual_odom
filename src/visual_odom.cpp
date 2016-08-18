#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
#include <Eigen/Geometry> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

using namespace message_filters;

Eigen::Affine3d create_rotation_matrix(double ax, double ay, double az) {
  Eigen::Affine3d rx =
      Eigen::Affine3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Affine3d ry =
      Eigen::Affine3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Affine3d rz =
      Eigen::Affine3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}

cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
cv::Size subPixWinSize(10,10), winSize(31,31);
int max_count = 100;

void findPlotPoints(cv::Mat &grey, cv::Mat &prevGrey, std::vector<cv::Point2f> points[],
    cv::Mat &frame, cv::Scalar color)
{
  std::vector<uchar> status;
  std::vector<float> err;
  if (prevGrey.empty())
  {
    grey.copyTo(prevGrey);
  }
  cv::calcOpticalFlowPyrLK(
      prevGrey, grey, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);
  
  size_t i, k;
  for (i = k = 0; i < points[1].size(); i++)
  {
    if (!status[i]) continue;
    points[1][k++] = points[1][i];
    cv::circle(frame, points[1][i], 3, color, -1, 8);
    cv::line(frame, points[1][i], points[0][i], color, 1, 8, 0);
  }
  points[1].resize(k);
}

std::vector<cv::Point2f> lpoints[2], rpoints[2], lrpoints[2], rlpoints[2];
cv::Mat prevLGrey, prevRGrey;
bool needToInit = true;
void callback(const sensor_msgs::ImageConstPtr& left_image, const sensor_msgs::ImageConstPtr& right_image)
{
  cv::Mat frame_left = cv_bridge::toCvShare(left_image, sensor_msgs::image_encodings::BGR8)->image;
  cv::Mat frame_right = cv_bridge::toCvShare(right_image, sensor_msgs::image_encodings::BGR8)->image;

  cv::namedWindow("Left debug", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Right debug", cv::WINDOW_AUTOSIZE);

  cv::Mat lgrey, rgrey;
  cv::cvtColor(frame_left, lgrey, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame_right, rgrey, cv::COLOR_BGR2GRAY);

  if (needToInit)
  {
    cv::goodFeaturesToTrack(lgrey, lpoints[1], max_count, 0.01, 10, cv::Mat(), 3, 0, 0.04);
    cv::cornerSubPix(lgrey, lpoints[1], subPixWinSize, cv::Size(-1,-1), termcrit);

    cv::goodFeaturesToTrack(rgrey, rpoints[1], max_count, 0.01, 10, cv::Mat(), 3, 0, 0.04);
    cv::cornerSubPix(rgrey, rpoints[1], subPixWinSize, cv::Size(-1,-1), termcrit);

    lrpoints[1] = lpoints[1];
    rlpoints[1] = rpoints[1];
  }
  else if(!lpoints[0].empty() && !rpoints[0].empty())
  {
    findPlotPoints(lgrey, prevLGrey, lpoints, frame_left, cv::Scalar(0, 255, 0));
    findPlotPoints(rgrey, prevRGrey, rpoints, frame_right, cv::Scalar(255, 0, 0));

    lrpoints[0] = lpoints[1];
    rlpoints[0] = rpoints[1];

    findPlotPoints(lgrey, rgrey, rlpoints, frame_left, cv::Scalar(255, 0, 0));
    findPlotPoints(rgrey, lgrey, lrpoints, frame_right, cv::Scalar(0, 255, 0));

    // lpoints and rpoints contain the rotation information
    // rlpoints and lrpoints contain the 3d information for the current points
    // Need to compare the keyframe 3d points to the current 3d points
    // First apply the rotation then average the translation
  }

  needToInit = false;
  cv::imshow("Left debug", frame_left);
  cv::imshow("Right debug", frame_right);
  cv::waitKey(1);
  std::swap(lpoints[1], lpoints[0]);
  cv::swap(prevLGrey, lgrey);

  std::swap(rpoints[1], rpoints[0]);
  cv::swap(prevRGrey, rgrey);
  
  std::swap(lrpoints[1], lrpoints[0]);
  std::swap(rlpoints[1], rlpoints[0]);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "visual_node");

  ros::NodeHandle nh;

  message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, "/usb_cam/left/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, "/usb_cam/right/image_raw", 1);

  typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ImageSyncPolicy;
  Synchronizer<ImageSyncPolicy> sync(ImageSyncPolicy(10), left_sub, right_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();

  return 0;
}

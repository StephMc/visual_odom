#include <visual_odom/camera_model.h>

void CameraModel::correctRadial(std::vector<cv::Point2f> &input,
    std::vector<cv::Point2f> &output, Camera cam)
{
  CameraParams cp = cam == LEFT_CAMERA ? l_params : r_params;
  for (int i = 0; i < input.size(); ++i)
  {
    double x = (input[i].x - cp.cx) / cp.f;
    double y = (input[i].y - cp.cy) / cp.f;
    double r2 = x * x + y * y;
    output[i].x = (x * (1 - cp.k1 * r2 - cp.k2 * r2 * r2)) * cp.f + cp.cx;
    output[i].y = (y * (1 - cp.k1 * r2 - cp.k2 * r2 * r2)) * cp.f + cp.cy;
  }
}

// Takes in points from the right camera and corrects right camera's
// rotation error.
void CameraModel::correctRotation(std::vector<cv::Point2f> &input,
    std::vector<cv::Point2f> &output)
{
  Eigen::Matrix3d rotationMat =
    create_rotation_matrix(-r_params.rx, -r_params.ry, -r_params.rz);
  for (int i = 0; i < input.size(); ++i)
  {
    Eigen::Vector3d p(input[i].x, input[i].y, r_params.f);
    p = rotationMat * p;
    output[i].x = p(0);
    output[i].y = p(1);
  }
}

void CameraModel::calculate3dPoints(
    std::vector<Eigen::Vector4d> &points3d,
    std::vector<cv::Point2f> lpoints, std::vector<cv::Point2f> rpoints)
{
  double focal = (l_params.f + r_params.f) / 2;
  double fb = focal * r_params.b;
  double cx = (l_params.cx + r_params.cx) / 2;
  double cy = (l_params.cy + r_params.cy) / 2;
  for (int i = 0; i < lpoints.size(); ++i)
  {
    Eigen::Vector4d p;
    p(0) = fb / (lpoints[i].x - rpoints[i].x);
    p(1) = -((lpoints[i].x - cx) * p(0)) / focal; 
    p(2) = -((lpoints[i].y - cy) * p(0)) / focal;
    p(3) = 1;
    points3d[i] = p;
  }
}

CameraModel::CameraParams CameraModel::getAverageCamera()
{
  return CameraParams(0, 0,
      (l_params.cx + r_params.cx) / 2, (l_params.cy + r_params.cy) / 2,
      (l_params.f + r_params.f) / 2, 0, 0, 0, 0);
}

Eigen::Matrix3d CameraModel::create_rotation_matrix(double ax, double ay, double az) {
  Eigen::Matrix3d rx =
      Eigen::Matrix3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Matrix3d ry =
      Eigen::Matrix3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Matrix3d rz =
      Eigen::Matrix3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}


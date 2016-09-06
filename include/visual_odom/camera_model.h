#ifndef CAMERA_MODEL_H
#define CAMERA_MODEL_H

#include <vector>
#include <Eigen/Geometry> 
#include <opencv2/opencv.hpp>

class CameraModel
{
public:
  struct CameraParams
  {
    CameraParams(double k1_, double k2_, double cx_, double cy_,
        double f_, double b_, double rx_, double ry_, double rz_) :
        k1(k1_), k2(k2_), cx(cx_), cy(cy_), f(f_), b(b_),
        rx(rx_), ry(ry_), rz(rz_)
    {
    }
    double k1, k2, cx, cy, f, b, rx, ry, rz;
  };
  typedef struct CameraParams CameraParams;

  enum Camera {LEFT_CAMERA, RIGHT_CAMERA};

  CameraModel(CameraParams left, CameraParams right) :
    l_params(left), r_params(right)
  {
  }

  void correctRadial(std::vector<cv::Point2f> &input,
      std::vector<cv::Point2f> &output, Camera cam);

  void correctRotation(std::vector<cv::Point2f> &input,
      std::vector<cv::Point2f> &output);

  void calculate3dPoints(std::vector<Eigen::Vector4d> &points3d,
      std::vector<cv::Point2f> lpoints,
      std::vector<cv::Point2f> rpoints);

  CameraParams getAverageCamera();

private:
  Eigen::Matrix3d create_rotation_matrix(double ax, double ay, double az);
  CameraParams l_params, r_params;
};
#endif

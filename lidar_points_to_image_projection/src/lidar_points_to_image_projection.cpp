#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

struct LidarPoint {
    double x, y, z, r;
};

void LoadLidarPoints(const std::string file_name,
                     std::vector<LidarPoint> &lidar_points) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_name << std::endl;
        return;
    }

    long size;
    file.read(reinterpret_cast<char *>(&size), sizeof(size));

    lidar_points.clear();
    for (int i = 0; i < size; ++i) {
        LidarPoint lidar_point;
        file.read(reinterpret_cast<char *>(&lidar_point), sizeof(LidarPoint));
        lidar_points.push_back(lidar_point);
    }

    file.close();
}

void LoadCalibrationData(cv::Mat &transformation_lidar_to_coplanar,
                         cv::Mat &transformation_coplanar_to_camera,
                         cv::Mat &intrinsic_camera_to_image) {
    transformation_lidar_to_coplanar.at<double>(0, 0) = 7.533745e-03;
    transformation_lidar_to_coplanar.at<double>(0, 1) = -9.999714e-01;
    transformation_lidar_to_coplanar.at<double>(0, 2) = -6.166020e-04;
    transformation_lidar_to_coplanar.at<double>(0, 3) = -4.069766e-03;
    transformation_lidar_to_coplanar.at<double>(1, 0) = 1.480249e-02;
    transformation_lidar_to_coplanar.at<double>(1, 1) = 7.280733e-04;
    transformation_lidar_to_coplanar.at<double>(1, 2) = -9.998902e-01;
    transformation_lidar_to_coplanar.at<double>(1, 3) = -7.631618e-02;
    transformation_lidar_to_coplanar.at<double>(2, 0) = 9.998621e-01;
    transformation_lidar_to_coplanar.at<double>(2, 1) = 7.523790e-03;
    transformation_lidar_to_coplanar.at<double>(2, 2) = 1.480755e-02;
    transformation_lidar_to_coplanar.at<double>(2, 3) = -2.717806e-01;
    transformation_lidar_to_coplanar.at<double>(3, 0) = 0.0;
    transformation_lidar_to_coplanar.at<double>(3, 1) = 0.0;
    transformation_lidar_to_coplanar.at<double>(3, 2) = 0.0;
    transformation_lidar_to_coplanar.at<double>(3, 3) = 1.0;

    transformation_coplanar_to_camera.at<double>(0, 0) = 9.999239e-01;
    transformation_coplanar_to_camera.at<double>(0, 1) = 9.837760e-03;
    transformation_coplanar_to_camera.at<double>(0, 2) = -7.445048e-03;
    transformation_coplanar_to_camera.at<double>(0, 3) = 0.0;
    transformation_coplanar_to_camera.at<double>(1, 0) = -9.869795e-03;
    transformation_coplanar_to_camera.at<double>(1, 1) = 9.999421e-01;
    transformation_coplanar_to_camera.at<double>(1, 2) = -4.278459e-03;
    transformation_coplanar_to_camera.at<double>(1, 3) = 0.0;
    transformation_coplanar_to_camera.at<double>(2, 0) = 7.402527e-03;
    transformation_coplanar_to_camera.at<double>(2, 1) = 4.351614e-03;
    transformation_coplanar_to_camera.at<double>(2, 2) = 9.999631e-01;
    transformation_coplanar_to_camera.at<double>(2, 3) = 0.0;
    transformation_coplanar_to_camera.at<double>(3, 0) = 0.0;
    transformation_coplanar_to_camera.at<double>(3, 1) = 0.0;
    transformation_coplanar_to_camera.at<double>(3, 2) = 0.0;
    transformation_coplanar_to_camera.at<double>(3, 3) = 1.0;

    intrinsic_camera_to_image.at<double>(0, 0) = 7.215377e+02;
    intrinsic_camera_to_image.at<double>(0, 1) = 0.000000e+00;
    intrinsic_camera_to_image.at<double>(0, 2) = 6.095593e+02;
    intrinsic_camera_to_image.at<double>(0, 3) = 0.000000e+00;
    intrinsic_camera_to_image.at<double>(1, 0) = 0.000000e+00;
    intrinsic_camera_to_image.at<double>(1, 1) = 7.215377e+02;
    intrinsic_camera_to_image.at<double>(1, 2) = 1.728540e+02;
    intrinsic_camera_to_image.at<double>(1, 3) = 0.000000e+00;
    intrinsic_camera_to_image.at<double>(2, 0) = 0.000000e+00;
    intrinsic_camera_to_image.at<double>(2, 1) = 0.000000e+00;
    intrinsic_camera_to_image.at<double>(2, 2) = 1.000000e+00;
    intrinsic_camera_to_image.at<double>(2, 3) = 0.000000e+00;
}

void ProjectLidarPointsToImage(const std::vector<LidarPoint> &lidar_points,
                               const cv::Mat &image) {
    // load calibration data
    cv::Mat transformation_lidar_to_coplanar(4, 4, cv::DataType<double>::type);
    cv::Mat transformation_coplanar_to_camera(4, 4, cv::DataType<double>::type);
    cv::Mat intrinsic_camera_to_image(3, 4, cv::DataType<double>::type);
    LoadCalibrationData(transformation_lidar_to_coplanar,
                        transformation_coplanar_to_camera,
                        intrinsic_camera_to_image);

    cv::Mat visualized_image = image.clone();
    for (const auto &lidar_point : lidar_points) {
        // focus on ego lane and high reflectivity
        double max_x = 20.0, max_y = 6.0, min_z = -1.4, min_r = 0.01;
        if (lidar_point.x > max_x || lidar_point.x < 0.0) {
            continue;
        }
        if (std::abs(lidar_point.y) > max_y) {
            continue;
        }
        if (lidar_point.z < min_z) {
            continue;
        }
        if (lidar_point.r < min_r) {
            continue;
        }

        // project lidar points to image
        cv::Mat X(4, 1, cv::DataType<double>::type);
        cv::Mat Y(3, 1, cv::DataType<double>::type);

        X.at<double>(0, 0) = lidar_point.x;
        X.at<double>(1, 0) = lidar_point.y;
        X.at<double>(2, 0) = lidar_point.z;
        X.at<double>(3, 0) = 1;
        Y = intrinsic_camera_to_image * transformation_coplanar_to_camera *
            transformation_lidar_to_coplanar * X;
        cv::Point point(Y.at<double>(0, 0) / Y.at<double>(2, 0),
                        Y.at<double>(1, 0) / Y.at<double>(2, 0));

        // visualize lidar points
        double max_value = 20.0;
        int red = std::min(
            255, static_cast<int>(
                     255 * std::abs((max_value - lidar_point.x) / max_value)));
        int green = std::min(
            255,
            static_cast<int>(
                255 * (1 - std::abs((max_value - lidar_point.x) / max_value))));

        cv::circle(visualized_image, point, 1, cv::Scalar(0, green, red), -1);
    }

    std::string window_name = "lidar points to image projection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, visualized_image);
    cv::waitKey(0);
}

int main() {
    // load image from file
    cv::Mat image = cv::imread("../images/0000000000.png");

    // load lidar points from file
    std::vector<LidarPoint> lidar_points;
    LoadLidarPoints("../point_clouds/C51_LidarPts_0000.dat", lidar_points);

    // project lidar points to image
    ProjectLidarPointsToImage(lidar_points, image);

    return 0;
}
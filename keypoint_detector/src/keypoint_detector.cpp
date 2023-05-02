#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/xfeatures2d/nonfree.hpp>

enum class DetectorType { SIFT, BRISK, ORB };

void DetectKeypoints(const cv::Mat &input_img,
                     const DetectorType &detector_type,
                     std::vector<cv::KeyPoint> &keypoints) {
    cv::Ptr<cv::FeatureDetector> detector;
    switch (detector_type) {
        case DetectorType::SIFT:
            detector = cv::SIFT::create();
            break;
        case DetectorType::BRISK:
            detector = cv::BRISK::create();
            break;
        case DetectorType::ORB:
            detector = cv::ORB::create();
            break;
    }

    detector->detect(input_img, keypoints);
}

int main() {
    // load image
    cv::Mat img = cv::imread("../images/img.png");
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    // detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    DetectKeypoints(gray_img, DetectorType::ORB, keypoints);

    // visualize result
    cv::Mat keypoints_img = gray_img.clone();
    cv::drawKeypoints(gray_img, keypoints, keypoints_img,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::string window_name = "keypoint detector";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, keypoints_img);
    cv::waitKey(0);

    return 0;
}
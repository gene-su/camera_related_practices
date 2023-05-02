#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/xfeatures2d/nonfree.hpp>

enum class DetectorType { SIFT, BRISK, ORB };
enum class ExtractorType { SIFT, BRISK, ORB };

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

void DescribeKeypoints(const cv::Mat &input_img,
                       const ExtractorType &extractor_type,
                       std::vector<cv::KeyPoint> &keypoints,
                       cv::Mat &descriptors) {
    cv::Ptr<cv::DescriptorExtractor> extractor;
    switch (extractor_type) {
        case ExtractorType::SIFT:
            extractor = cv::SIFT::create();
            break;
        case ExtractorType::BRISK:
            extractor = cv::BRISK::create();
            break;
        case ExtractorType::ORB:
            extractor = cv::ORB::create();
            break;
    }

    extractor->compute(input_img, keypoints, descriptors);
}

int main() {
    // load image
    cv::Mat img = cv::imread("../images/img.png");
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    // detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    DetectKeypoints(gray_img, DetectorType::ORB, keypoints);

    // describe keypoints
    cv::Mat descriptors;
    DescribeKeypoints(gray_img, ExtractorType::ORB, keypoints, descriptors);

    std::cout << descriptors.size() << std::endl;
    for (int col = 0; col < descriptors.cols; ++col) {
        std::cout << static_cast<int>(descriptors.at<uchar>(0, col)) << ", ";
    }
    std::cout << std::endl;

    return 0;
}
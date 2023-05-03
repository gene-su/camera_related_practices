#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/xfeatures2d/nonfree.hpp>

enum class DetectorType { SIFT, BRISK, ORB };
enum class ExtractorType { SIFT, BRISK, ORB };
enum class MatcherType { BF, FLANN };
enum class SelectorType { NN, KNN };

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

void MatchKeypoints(const ExtractorType &extractor_type,
                    const MatcherType &matcher_type,
                    const SelectorType &selector_type,
                    cv::Mat &prev_descriptors, cv::Mat &curr_descriptors,
                    std::vector<cv::DMatch> &matches) {
    cv::Ptr<cv::DescriptorMatcher> matcher;
    switch (matcher_type) {
        case MatcherType::BF:
            switch (extractor_type) {
                case ExtractorType::SIFT:
                    matcher = cv::DescriptorMatcher::create(
                        cv::DescriptorMatcher::BRUTEFORCE);
                    break;
                default:
                    matcher = cv::DescriptorMatcher::create(
                        cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
                    break;
            }
            break;
        case MatcherType::FLANN:
            switch (extractor_type) {
                case ExtractorType::SIFT:
                    matcher = cv::DescriptorMatcher::create(
                        cv::DescriptorMatcher::FLANNBASED);
                    break;
                default:
                    matcher = cv::makePtr<cv::FlannBasedMatcher>(
                        cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
                    break;
            }
            break;
    }

    switch (selector_type) {
        case SelectorType::NN:
            matcher->match(prev_descriptors, curr_descriptors, matches);
            break;
        case SelectorType::KNN:
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(prev_descriptors, curr_descriptors, knn_matches,
                              2);
            double min_distance_ratio = 0.8;
            for (const auto &knn_match : knn_matches) {
                if (knn_match.size() == 2) {
                    if (knn_match.at(0).distance <
                        knn_match.at(1).distance * min_distance_ratio) {
                        matches.push_back(knn_match.at(0));
                    }
                }
            }
            break;
    }
}

int main() {
    // load image
    cv::Mat prev_img = cv::imread("../images/0000000000.png");
    cv::cvtColor(prev_img, prev_img, cv::COLOR_BGR2GRAY);
    cv::Mat curr_img = cv::imread("../images/0000000001.png");
    cv::cvtColor(curr_img, curr_img, cv::COLOR_BGR2GRAY);

    // detect keypoints
    std::vector<cv::KeyPoint> prev_keypoints;
    DetectKeypoints(prev_img, DetectorType::ORB, prev_keypoints);
    std::vector<cv::KeyPoint> curr_keypoints;
    DetectKeypoints(curr_img, DetectorType::ORB, curr_keypoints);

    // describe keypoints
    cv::Mat prev_descriptors;
    DescribeKeypoints(prev_img, ExtractorType::ORB, prev_keypoints,
                      prev_descriptors);
    cv::Mat curr_descriptors;
    DescribeKeypoints(curr_img, ExtractorType::ORB, curr_keypoints,
                      curr_descriptors);

    // match keypoints
    std::vector<cv::DMatch> matches;
    MatchKeypoints(ExtractorType::ORB, MatcherType::FLANN, SelectorType::KNN,
                   prev_descriptors, curr_descriptors, matches);             

    // visualize result
    cv::Mat matched_keypoints_img;
    cv::drawMatches(prev_img, prev_keypoints, curr_img, curr_keypoints, matches,
                    matched_keypoints_img, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::string window_name = "keypoint matcher";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, matched_keypoints_img);
    cv::waitKey(0);

    return 0;
}
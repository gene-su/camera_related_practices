#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_set>

void GetCornerResponseMap(const cv::Mat& input_img, cv::Mat& output_img) {
    int block_size = 2;     // window size
    int aperture_size = 3;  // Sobel filter size
    double k = 0.04;

    // convert to grayscale
    cv::Mat gray_img;
    cv::cvtColor(input_img, gray_img, cv::COLOR_BGR2GRAY);

    // construct corner response map
    cv::Mat corner_response_map;
    cv::cornerHarris(gray_img, corner_response_map, block_size, aperture_size,
                     k,
                     cv::BORDER_DEFAULT);  // corner response map (CV_32F)
    cv::normalize(corner_response_map, corner_response_map, 0, 255,
                  cv::NORM_MINMAX);  // normalize to 0.~255. (CV_32F)

    output_img = corner_response_map;
}

std::vector<cv::KeyPoint> GetHighResponseKeypoints(const cv::Mat& input_img) {
    float min_response = 100.;

    std::vector<cv::KeyPoint> keypoints;
    for (int row = 0; row < input_img.rows; ++row) {
        for (int col = 0; col < input_img.cols; ++col) {
            if (input_img.at<float>(row, col) > min_response) {
                cv::KeyPoint keypoint;
                keypoint.pt = cv::Point2f(col, row);
                keypoint.size = 5;
                keypoint.response = input_img.at<float>(row, col);

                keypoints.push_back(keypoint);
            }
        }
    }

    return keypoints;
}

std::vector<cv::KeyPoint> GetNmsHighResponseKeypoints(
    const std::vector<cv::KeyPoint>& keypoints) {
    double max_overlap = 0.0;

    std::vector<cv::KeyPoint> sorted_keypoints = keypoints;
    std::sort(sorted_keypoints.begin(), sorted_keypoints.end(),
              [](const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) {
                  return kp1.response > kp2.response;
              });

    std::unordered_set<int> visited_keypoints;
    std::vector<cv::KeyPoint> nms_keypoins;
    for (std::size_t i = 0; i < sorted_keypoints.size(); ++i) {
        if (visited_keypoints.find(i) != visited_keypoints.end()) {
            continue;
        }
        for (std::size_t j = i + 1; j < sorted_keypoints.size(); ++j) {
            if (visited_keypoints.find(j) != visited_keypoints.end()) {
                continue;
            }

            if (cv::KeyPoint::overlap(sorted_keypoints[i],
                                      sorted_keypoints[j]) > max_overlap) {
                visited_keypoints.insert(j);
            }
        }
        nms_keypoins.push_back(sorted_keypoints[i]);
    }

    return nms_keypoins;
}

int main() {
    // load image
    cv::Mat img = cv::imread("../images/img.png");

    // compute corner response map
    cv::Mat corner_response_map;
    GetCornerResponseMap(img, corner_response_map);

    cv::Mat visualized_corner_response_map;
    cv::convertScaleAbs(corner_response_map,
                        visualized_corner_response_map);  // CV_32F to CV_8U
    std::string window_name = "harris corner";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, visualized_corner_response_map);
    cv::waitKey(0);
    cv::imwrite("output.png", visualized_corner_response_map);

    // get high response keypoint
    std::vector<cv::KeyPoint> high_response_keypoints =
        GetHighResponseKeypoints(corner_response_map);

    cv::Mat harris_corner = visualized_corner_response_map.clone();
    cv::drawKeypoints(visualized_corner_response_map, high_response_keypoints,
                      harris_corner, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(window_name, harris_corner);
    cv::waitKey(0);

    // get nms high response keypoint
    std::vector<cv::KeyPoint> nms_high_response_keypoints =
        GetNmsHighResponseKeypoints(high_response_keypoints);

    cv::Mat nms_harris_corner = visualized_corner_response_map.clone();
    cv::drawKeypoints(visualized_corner_response_map,
                      nms_high_response_keypoints, nms_harris_corner,
                      cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(window_name, nms_harris_corner);
    cv::waitKey(0);

    return 0;
}
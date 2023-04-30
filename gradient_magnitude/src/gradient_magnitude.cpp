#include <array>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void GradientMagnitude(const cv::Mat& input_img, cv::Mat& output_img) {
    // convert to grayscale
    cv::Mat gray_img;
    cv::cvtColor(input_img, gray_img, cv::COLOR_BGR2GRAY);

    // gaussian smoothing
    cv::Mat blurred_img;
    int kernel_size = 5;
    int standard_deviation = 2.0;
    cv::GaussianBlur(gray_img, blurred_img, cv::Size(kernel_size, kernel_size),
                     standard_deviation);

    // sobel operator
    std::array<float, 9> sobel_x = {-1., 0., 1., -2., 0., 2., -1., 0., 1.};
    std::array<float, 9> sobel_y = {-1., -2., -1., 0., 0., 0., 1., 2., 1.};
    cv::Mat kernel_x{cv::Size{3, 3}, CV_32F, sobel_x.data()};
    cv::Mat kernel_y{cv::Size{3, 3}, CV_32F, sobel_y.data()};

    cv::Mat gradient_x, gradient_y;
    cv::filter2D(blurred_img, gradient_x, -1, kernel_x, cv::Point(-1, -1), 0,
                 cv::BORDER_DEFAULT);
    cv::filter2D(blurred_img, gradient_y, -1, kernel_y, cv::Point(-1, -1), 0,
                 cv::BORDER_DEFAULT);

    // gradient magnitude
    cv::Mat gradient_magnitude = gray_img.clone();
    for (int row = 0; row < gradient_magnitude.rows; ++row) {
        for (int col = 0; col < gradient_magnitude.cols; ++col) {
            gradient_magnitude.at<unsigned char>(row, col) =
                std::sqrt(std::pow(gradient_x.at<unsigned char>(row, col), 2) +
                          std::pow(gradient_y.at<unsigned char>(row, col), 2));
        }
    }
    
    output_img = gradient_magnitude;
}

int main() {
    // load image
    cv::Mat img = cv::imread("../images/img.png");

    // compute gradient magnitude
    cv::Mat gradient_magnitude;
    GradientMagnitude(img, gradient_magnitude);

    // show result
    std::string window_name = "gradient magnitude";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, gradient_magnitude);
    cv::waitKey(0);

    return 0;
}
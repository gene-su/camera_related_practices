#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

struct Object {
    std::string name;
    double score;
    cv::Rect box;
};

void DetectObjects(const cv::Mat& image) {
    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("../yolo/yolov3.cfg",
                                                   "../yolo/yolov3.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // get class names
    std::vector<std::string> class_names;
    std::ifstream file("../yolo/coco.names");
    std::string line;
    while (std::getline(file, line)) {
        class_names.push_back(line);
    }

    // get input from input image
    cv::Mat input;
    cv::dnn::blobFromImage(image, input, 1 / 255.0, cv::Size(416, 416),
                           cv::Scalar(0, 0, 0), false, false);

    // get outputs
    std::vector<cv::String> output_layers = net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;
    net.setInput(input);
    net.forward(outputs, output_layers);
    // outputs: [[85 x 507], [85 x 2028], [85 x 8112]]
    // 85: 4 (x,y,w,h), 1 confidence, 80 classes scores

    // outputs to classes, max_scores, boxes
    double score_threshold = 0.2;
    std::vector<std::string> classes;
    std::vector<float> max_socres;
    std::vector<cv::Rect> boxes;
    for (size_t i = 0; i < outputs.size(); ++i) {
        cv::Mat detection = outputs[i];
        for (int j = 0; j < detection.rows; ++j) {
            cv::Mat scores = detection.row(j).colRange(5, detection.cols);
            cv::Point max_score_id;
            double max_score;
            cv::minMaxLoc(scores, 0, &max_score, 0, &max_score_id);

            if (max_score > score_threshold) {
                int center_x =
                    static_cast<int>(detection.at<float>(j, 0) * image.cols);
                int center_y =
                    static_cast<int>(detection.at<float>(j, 1) * image.rows);
                int width =
                    static_cast<int>(detection.at<float>(j, 2) * image.cols);
                int height =
                    static_cast<int>(detection.at<float>(j, 3) * image.rows);
                int left = center_x - width / 2;
                int top = center_y - height / 2;

                classes.push_back(class_names.at(max_score_id.x));
                max_socres.push_back(static_cast<float>(max_score));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // perform non-maxima suppression
    double num_threshold = 0.4;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, max_socres, score_threshold, num_threshold,
                      indices);

    // get objects
    std::vector<Object> objects;
    for (const auto& index : indices) {
        Object object;
        object.name = classes.at(index);
        object.score = max_socres.at(index);
        object.box = boxes.at(index);
        objects.push_back(object);
    }

    // visualize result
    cv::Mat object_detection_image = image.clone();
    for (const auto& object : objects) {
        // draw box
        cv::rectangle(object_detection_image,
                      cv::Point(object.box.x, object.box.y),
                      cv::Point(object.box.x + object.box.width,
                                object.box.y + object.box.height),
                      cv::Scalar(0, 255, 0), 2);

        // draw label
        std::string label =
            object.name + ": " + cv::format("%.2f", object.score);

        cv::Size label_size =
            getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 0, nullptr);
        cv::rectangle(object_detection_image,
                      cv::Point(object.box.x, object.box.y - label_size.height),
                      cv::Point(object.box.x + label_size.width, object.box.y),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(object_detection_image, label,
                    cv::Point(object.box.x, object.box.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 0);
    }

    std::string window_name = "object detection";
    cv::namedWindow(window_name, 1);
    cv::imshow(window_name, object_detection_image);
    cv::waitKey(0);
}

int main() {
    // load image
    cv::Mat image = cv::imread("../images/img.png");

    // detect objects
    DetectObjects(image);

    return 0;
}
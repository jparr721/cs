#include "slam.h"

namespace slam {
  void Slam::process_frame(std::string frame_name) {
    std::cout << "Opening frame: " << frame_name << std::endl;
    cv::VideoCapture cap(frame_name);

    if (!cap.isOpened()) {
      std::cout << "Error opening the video stream" << std::endl;
      return;
    }
    for (;;) {
      cv::Mat frame;
      cap >> frame;

      if (frame.empty()) {
        break;
      }

      imshow("Frame", frame);

      char c = (char) cv::waitKey(25);
      if (c == 27) {
        break;
      }
    }

    cap.release();
    cv::destroyAllWindows();
  }

  void Slam::detect_features(cv::Mat img, std::vector<cv::Point2f>& points) {
    std::vector<cv::KeyPoint> keypoints;
    // Fast threshold of 20 and nonmax suppression enabled
    cv::FAST(img, keypoints, 20, true);
    cv::KeyPoint::convert(keypoints, points, std::vector<int>());
  }

  void Slam::track_features(
      cv::Mat img1,
      cv::Mat img2,
      std::vector<cv::Point2f>& points1,
      std::vector<cv::Point2f>& points2,
      std::vector<uchar>& status) {
    std::vector<float> error;
    cv::Size window_size = cv::Size(21, 21);
    cv::TermCriteria term_crit = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFLowPyrLK(
        img1, img2, points1, ponts2, status, error, window_size, 3, term_crit, 0, 0.001);

    // Get rid of points that the KLT tracker couldn't track
    int index_correction = 0;
    for (int i = 0; i < status.size(); ++i) {
      cv::Point2f t = points2.at(1 - index_correction);
      if (status.at(i) == 0 || pt.x < 0 || pt.y < 0) {
        if (pt.x < 0 || pt.y < 0) {
          status.at(i) = 0;
        }
        points1.erase(points1.begin() + i - index_correction);
        points2.erase(points2.begin() + i - index_correction);
        ++index_correction;
      }
    }
  }
} // namespace slam

int main(int argc, char** argv) {
  slam::Slam s;

  s.process_frame(argv[1]);

  return EXIT_SUCCESS;
}

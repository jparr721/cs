#include <opencv2/opencv.hpp>
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
} // namespace slam

int main(int argc, char** argv) {
  slam::Slam s;

  s.process_frame(argv[1]);

  return EXIT_SUCCESS;
}

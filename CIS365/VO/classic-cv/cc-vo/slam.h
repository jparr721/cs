#ifndef SLAM_H_
#define SLAM_H_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace slam {
  class Slam {
    public:
      void process_frame();
      void detect_features(cv::Mat img, std::vector<cv::Point2f>& points);
      void track_features(
          cv::Mat img1,
          cv::Mat img2,
          std::vector<cv::Point2f>& points1,
          std::vector<cv::Point2f>& points2,
          std::vector<uchar>& status);

      double get_absolute_scale(int frame_id, int sequence_id, double z_cal);

      static constexpr int MAX_FRAME = 1000;
      static constexpr int MIN_NUM_FEAT = 2000;
  };
} // namespace slam

#endif // SLAM_H_

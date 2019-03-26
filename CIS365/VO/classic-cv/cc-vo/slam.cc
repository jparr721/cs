#include <algorithm>
#include <cmath>
#include <ctime>
#include <ctype.h>
#include <iterator>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "slam.h"

namespace slam {
  void Slam::process_frame() {
    // Our rotation and translation vectors
    cv::Mat R_f;
    cv::Mat t_f;

    // Load our results into memory
    std::ofstream resultsdata("./results_1.txt");

    int font_face{1};
    int thickness{1};

    double scale{1.0};
    double font_scale{1.0};
    char file1[100];
    char file2[100];
    std::sprintf(file1, "../../../dataset/sequences/00/image_0/%06d.png", 0);
    std::sprintf(file2, "../../../dataset/sequences/00/image_0/%06d.png", 1);

    cv::Point text_org(10, 50);

    // Read the first two frames from the dataset
    cv::Mat img1 = cv::imread(file1);
    cv::Mat img2 = cv::imread(file2);

    // Check if it blew up
    if (!img1.data || !img2.data) { std::cerr << "Failed to read image" << std::endl; }

    // Begin our feature detection algorithm
    std::vector<cv::Point2f> points1, points2;

    // Store features into our point vector
    detect_features(img1, points1);
    std::vector<uchar> status;
    // Track our features between image one and two in a sequence
    track_features(img1, img2, points1, points2, status);

    // TODO(jparr721): Ensure these focal values are correct
    // Ideally these should load from the calibration files
    double focal{718.8560};
    cv::Point2d pp(607.1928, 185.2157);

    // Get the pose and essential matrix via y.T * E * y quadratic form transformation
    cv::Mat E, R, t, mask;

    // Find this via the RANSAC algorithm
    // E is our essential matrix
    E = cv::findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);

    // Recover the relative camera rotation and translation from the essential matrix
    cv::recoverPose(E, points2, points1, R, t, focal, pp, mask);

    cv::Mat prev_image = img2;
    cv::Mat cur_image;
    std::vector<cv::Point2f> prev_features = points2;
    std::vector<cv::Point2f> cur_features;

    char filename[100];
    char text[100];

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();

    cv::namedWindow("Road facing camera", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    cv::Mat traj = cv::Mat::zeros(cv::Size(600, 600), CV_8UC3);

    for (int i = 2; i < MAX_FRAME; ++i) {
      std::vector<uchar> status;
      std::sprintf(filename, "../../../dataset/sequences/00/image_0/%06d.png", i);

      cv::Mat cur_image = cv::imread(filename);

      track_features(prev_image, cur_image, prev_features, cur_features, status);

      E = cv::findEssentialMat(cur_features, prev_features, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
      cv::recoverPose(E, cur_features, prev_features, R, t, focal, pp, mask);

      cv::Mat prev_points(2, prev_features.size(), CV_64F);
      cv::Mat cur_points(2, cur_features.size(), CV_64F);

      for (int j = 0; i < prev_features.size(); ++i) {
        prev_points.at<double>(0, j) = prev_features.at(j).x;
        prev_points.at<double>(1, j) = prev_features.at(j).y;

        cur_points.at<double>(0, j) = prev_features.at(j).x;
        cur_points.at<double>(1, j) = prev_features.at(j).y;
      }

      if (scale > 0.1 && t.at<double>(2) > t.at<double>(0) && t.at<double>(2) > t.at<double>(1)) {
        t_f = t_f + scale * (R_f * t);
        R_f = R*R_f;
      }

      if (prev_features.size() < MIN_NUM_FEAT) {
        detect_features(prev_image, prev_features);
        track_features(prev_image, cur_image, prev_features, cur_features, status);
      }

      prev_image = cur_image.clone();
      prev_features = cur_features;

      int x = int(t_f.at<double>(0) + 300);
      int y = int(t_f.at<double>(2) + 100);
      cv::circle(traj, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 2);
      cv::rectangle(traj, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);

      std::sprintf(text, "Coordinates: x = %02fm y=%02fm z= %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
      std::cout << text << std::endl;
      cv::putText(traj, text, text_org, font_face, font_scale, cv::Scalar::all(255), thickness, 8);

      cv::imshow("Road facing camera", cur_image);
      cv::imshow("Trajectory", traj);
      cv::waitKey(1);
    }

    clock_t end = clock();
    double elapsed{double(end - begin) / CLOCKS_PER_SEC};
    std::cout << "Total time taken: " << elapsed << std::endl;
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
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, error, window_size, 3, term_crit, 0, 0.001);

    // Get rid of points that the KLT tracker couldn't track or if they've gone out of frame
    int index_correction = 0;
    for (int i = 0; i < status.size(); ++i) {
      cv::Point2f t = points2.at(i - index_correction);
      if (status.at(i) == 0 || t.x < 0 || t.y < 0) {
        if (t.x < 0 || t.y < 0) {
          status.at(i) = 0;
        }
        points1.erase(points1.begin() + i - index_correction);
        points2.erase(points2.begin() + i - index_correction);
        ++index_correction;
      }
    }
  }

  double Slam::get_absolute_scale(int frame_id, int sequence_id, double z_cal) {
    int i = 0;

    double x{0.0};
    double y{0.0};
    double z{0.0};
    double x_prev{0.0};
    double y_prev{0.0};
    double z_prev{0.0};

    std::string line;

    std::ifstream datfile("../../../dataset/sequences/00/calib.txt");

    if (datfile.is_open()) {
      for(;(std::getline(datfile, line)) && (i <= frame_id);) {
        x_prev = x;
        y_prev = y;
        z_prev = z;

        std::stringstream in(line);
        for (int j = 0; j < 12; ++j) {
          in >> z;
          if (j == 7) y = z;
          if (j == 3) x = z;
        }

        ++i;
      }

      datfile.close();
    } else {
      std::cerr << "Failed to open file" << std::endl;
    }

    return std::sqrt(std::pow((x - x_prev), 2) + std::pow((y-y_prev), 2) + std::pow((z-z_prev), 2));
  }

} // namespace slam
